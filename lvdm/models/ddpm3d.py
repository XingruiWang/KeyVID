"""
wild mixture of
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/CompVis/taming-transformers
-- merci
"""
import os
from functools import partial
from contextlib import contextmanager
import numpy as np
from tqdm import tqdm
from einops import rearrange, repeat
import logging
mainlogger = logging.getLogger('mainlogger')
import random
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torchvision.utils import make_grid
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from utils.utils import instantiate_from_config
from lvdm.ema import LitEma
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.distributions import DiagonalGaussianDistribution
from lvdm.models.utils_diffusion import make_beta_schedule, rescale_zero_terminal_snr
from lvdm.basics import disabled_train
from lvdm.common import (
    extract_into_tensor,
    noise_like,
    exists,
    default
)

from imagebind import data
import torch
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType
import math


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def interpolate_pos_embed(pos_embed, model_pos_embed):
    cls_embed = pos_embed[:, 0, :]
    pos_embed = pos_embed[:, 1:, :]
    pos_embed = rearrange(pos_embed, 'b (q t) d -> b d q t', q=12)
    
    
    T = (model_pos_embed.size(1) - 1) // 12


    inter_pos_embed = torch.nn.functional.interpolate(
        pos_embed, size=(12, T), mode='bilinear', align_corners=False
    )


    inter_pos_embed = rearrange(inter_pos_embed, 'b d q t -> b (q t) d')
    inter_pos_embed = torch.cat((cls_embed.unsqueeze(1), inter_pos_embed), dim=1)

    assert inter_pos_embed.size() == model_pos_embed.size()
    return inter_pos_embed

class DDPM(pl.LightningModule):
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor=None,
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 rescale_betas_zero_snr=False,
                 ):
        super().__init__()
        assert parameterization in ["eps", "x0", "v"], 'currently only supporting "eps" and "x0" and "v"'
        self.parameterization = parameterization
        mainlogger.info(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.channels = channels
        self.temporal_length = unet_config.params.temporal_length
        self.image_size = image_size  # try conv?
        if isinstance(self.image_size, int):
            self.image_size = [self.image_size, self.image_size]
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        #count_params(self.model, verbose=True)
        self.use_ema = use_ema
        self.rescale_betas_zero_snr = rescale_betas_zero_snr
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            mainlogger.info(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        ## for reschedule
        self.given_betas = given_betas
        self.beta_schedule = beta_schedule
        self.timesteps = timesteps
        self.cosine_s = cosine_s

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        if self.rescale_betas_zero_snr:
            betas = rescale_zero_terminal_snr(betas)
        
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))

        if self.parameterization != 'v':
            self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        else:
            self.register_buffer('sqrt_recip_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.zeros_like(to_torch(alphas_cumprod)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        elif self.parameterization == "v":
            lvlb_weights = torch.ones_like(self.betas ** 2 / (
                    2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod)))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                mainlogger.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    mainlogger.info(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    mainlogger.info("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        mainlogger.info(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            mainlogger.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            mainlogger.info(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_start_from_z_and_v(self, x_t, t, v):
        # self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        # self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x_t.shape) * v +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * x_t
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_v(self, x, noise, t):
        return (
                extract_into_tensor(self.sqrt_alphas_cumprod, t, x.shape) * noise -
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * x
        )

    def get_loss(self, pred, target, mean=True):
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        # b, c, h, w, device, img_size, = *x.shape, x.device, self.image_size
        # assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        x = batch[k]
        '''
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        '''
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=True, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt

class LatentDiffusion(DDPM):
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 num_timesteps_cond=None,
                 cond_stage_key="caption",
                 audio_cond_key=None,
                 cond_stage_trainable=False,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 uncond_prob=0.2,
                 uncond_type="empty_seq",
                 scale_factor=1.0,
                 scale_by_std=False,
                 encoder_type="2d",
                 only_model=False,
                 noise_strength=0,
                 use_dynamic_rescale=False,
                 base_scale=0.7,
                 turning_step=400,
                 interp_mode=False,
                 fps_condition_type='fs',
                 perframe_ae=False,
                 use_qformer=True,
                 # added
                 logdir=None,
                 rand_cond_frame=False,
                 en_and_decode_n_samples_a_time=None,
                 *args, **kwargs):
        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        conditioning_key = default(conditioning_key, 'crossattn')
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)

        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key
        self.audio_cond_key = audio_cond_key
        self.noise_strength = noise_strength
        self.use_dynamic_rescale = use_dynamic_rescale
        self.interp_mode = interp_mode
        self.fps_condition_type = fps_condition_type
        self.perframe_ae = perframe_ae
        self.use_qformer = use_qformer
        self.logdir = logdir
        self.rand_cond_frame = rand_cond_frame
        self.en_and_decode_n_samples_a_time = en_and_decode_n_samples_a_time

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

        if use_dynamic_rescale:
            scale_arr1 = np.linspace(1.0, base_scale, turning_step)
            scale_arr2 = np.full(self.num_timesteps, base_scale)
            scale_arr = np.concatenate((scale_arr1, scale_arr2))
            to_torch = partial(torch.tensor, dtype=torch.float32)
            self.register_buffer('scale_arr', to_torch(scale_arr))

        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)
        self.first_stage_config = first_stage_config
        self.cond_stage_config = cond_stage_config        
        self.clip_denoised = False

        self.cond_stage_forward = cond_stage_forward
        self.encoder_type = encoder_type
        assert(encoder_type in ["2d", "3d"])
        self.uncond_prob = uncond_prob
        self.classifier_free_guidance = True if uncond_prob > 0 else False
        assert(uncond_type in ["zero_embed", "empty_seq"])
        self.uncond_type = uncond_type

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys, only_model=only_model)
            self.restarted_from_ckpt = True
                
    def make_cond_schedule(self, ):
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx=None):
        # only for very first batch, reset the self.scale_factor
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and \
                not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            mainlogger.info("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            mainlogger.info(f"setting self.scale_factor to {self.scale_factor}")
            mainlogger.info("### USING STD-RESCALING ###")
            mainlogger.info(f"std={z.flatten().std()}")

    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        if not self.cond_stage_trainable:
            model = instantiate_from_config(config)
            self.cond_stage_model = model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False
        else:
            model = instantiate_from_config(config)
            self.cond_stage_model = model
    
    def get_learned_conditioning(self, c):
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def get_first_stage_encoding(self, encoder_posterior, noise=None):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample(noise=noise)
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z
   
    @torch.no_grad()
    def encode_first_stage(self, x):
        if self.encoder_type == "2d" and x.dim() == 5:
            b, _, t, _, _ = x.shape
            x = rearrange(x, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
        
        ## consume more GPU memory but faster
        if not self.perframe_ae:
            encoder_posterior = self.first_stage_model.encode(x)
            results = self.get_first_stage_encoding(encoder_posterior).detach()
        else:  ## consume less GPU memory but slower
            results = []
            for index in range(x.shape[0]):
                frame_batch = self.first_stage_model.encode(x[index:index+1,:,:,:])
                frame_result = self.get_first_stage_encoding(frame_batch).detach()
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        
        return results
    
    def decode_core(self, z, **kwargs):
        if self.encoder_type == "2d" and z.dim() == 5:
            b, _, t, _, _ = z.shape
            z = rearrange(z, 'b c t h w -> (b t) c h w')
            reshape_back = True
        else:
            reshape_back = False
            
        if not self.perframe_ae:    
            z = 1. / self.scale_factor * z
            results = self.first_stage_model.decode(z, **kwargs)
        else:
            results = []
            for index in range(z.shape[0]):
                frame_z = 1. / self.scale_factor * z[index:index+1,:,:,:]
                frame_result = self.first_stage_model.decode(frame_z, **kwargs)
                results.append(frame_result)
            results = torch.cat(results, dim=0)

        if reshape_back:
            results = rearrange(results, '(b t) c h w -> b c t h w', b=b,t=t)
        return results

    @torch.no_grad()
    def decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, **kwargs):
        return self.decode_core(z, **kwargs)
    
    @torch.no_grad()
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, return_original_cond=False):
        ## video shape: b, c, t, h, w
        x = super().get_input(batch, self.first_stage_key)

        ## encode video frames x to z via a 2D encoder
        z = self.encode_first_stage(x)
                
        ## get caption condition
        cond = batch[self.cond_stage_key]
        if random_uncond and self.uncond_type == 'empty_seq':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond[i] = ""
        if isinstance(cond, dict) or isinstance(cond, list):
            cond_emb = self.get_learned_conditioning(cond)
        else:
            cond_emb = self.get_learned_conditioning(cond.to(self.device))
        if random_uncond and self.uncond_type == 'zero_embed':
            for i, ci in enumerate(cond):
                if random.random() < self.uncond_prob:
                    cond_emb[i] = torch.zeros_like(cond_emb[i])
        
        out = [z, cond_emb]


        ## optional output: self-reconst or caption
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond)

        return out

    def forward(self, x, c, **kwargs):
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.use_dynamic_rescale:
            x = x * extract_into_tensor(self.scale_arr, t, x.shape)
        return self.p_losses(x, c, t, **kwargs)

    def shared_step(self, batch, random_uncond, **kwargs):
        # Start here. take the batch
        x, c = self.get_batch_input(batch, random_uncond=random_uncond)
        loss, loss_dict = self(x, c, **kwargs)

        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, **kwargs):
        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        x_recon = self.model(x_noisy, t, **cond, **kwargs)
        
        if isinstance(x_recon, tuple):
            return x_recon[0]
        else:
            return x_recon

    def p_losses(self, x_start, cond, t, noise=None, **kwargs):
        if self.noise_strength > 0:
            b, c, f, _, _ = x_start.shape
            offset_noise = torch.randn(b, c, f, 1, 1, device=x_start.device)
            noise = default(noise, lambda: torch.randn_like(x_start) + self.noise_strength * offset_noise)
        else:
            noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        model_output = self.apply_model(x_noisy, t, cond, **kwargs)
        

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3, 4])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        if self.logvar.device is not self.device:
            self.logvar = self.logvar.to(self.device)
        logvar_t = self.logvar[t]
        # logvar_t = self.logvar[t.item()].to(self.device) # device conflict when ddp shared
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3, 4))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict  

    def training_step(self, batch, batch_idx):
        loss, loss_dict = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        ## sync_dist | rank_zero_only 
        self.log_dict(loss_dict, prog_bar=True, logger=True, on_step=True, on_epoch=True, sync_dist=False)
        #self.log("epoch/global_step", self.global_step.float(), prog_bar=True, logger=True, on_step=True, on_epoch=False)
        '''
        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False, rank_zero_only=True)
        '''
        if (batch_idx+1) % self.log_every_t == 0:
            mainlogger.info(f"batch:{batch_idx}|epoch:{self.current_epoch} [globalstep:{self.global_step}]: loss={loss}")
        return loss

    def validation_step(self, batch, batch_idx):
        _, loss_dict_no_ema = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch, random_uncond=self.classifier_free_guidance)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=True, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=True, on_epoch=True)
    
    def _get_denoise_row_from_list(self, samples, desc=''):
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device)))
        n_log_timesteps = len(denoise_row)

        denoise_row = torch.stack(denoise_row)  # n_log_timesteps, b, C, H, W
        
        if denoise_row.dim() == 5:
            denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
            denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=n_log_timesteps)
        elif denoise_row.dim() == 6:
            # video, grid_size=[n_log_timesteps*bs, t]
            video_length = denoise_row.shape[3]
            denoise_grid = rearrange(denoise_row, 'n b c t h w -> b n c t h w')
            denoise_grid = rearrange(denoise_grid, 'b n c t h w -> (b n) c t h w')
            denoise_grid = rearrange(denoise_grid, 'n c t h w -> (n t) c h w')
            denoise_grid = make_grid(denoise_grid, nrow=video_length)
        else:
            raise ValueError

        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=200, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, **kwargs):
        """ log images for LatentDiffusion """
        ##### control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 2
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        
        z, c, xrec, xc = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True)
        N = xrec.shape[0]
        log["reconst"] = xrec
        log["condition"] = xc
        

        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_cat, c_emb = c["c_concat"][0], c["c_crossattn"][0]
                    log["condition_cat"] = c_cat
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc = torch.zeros_like(c_emb)
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc]}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_x0=False, score_corrector=None, corrector_kwargs=None, **kwargs):
        t_in = t
        model_out = self.apply_model(x, t_in, c, **kwargs)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        if return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False, return_x0=False, \
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None, **kwargs):
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised, return_x0=return_x0, \
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs, **kwargs)
        if return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False, x_T=None, verbose=True, callback=None, \
                      timesteps=None, mask=None, x0=None, img_callback=None, start_T=None, log_every_t=None, **kwargs):
        # import ipdb; ipdb.set_trace()
        
        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]        
        # sample an initial noise
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps
        if start_T is not None:
            timesteps = min(timesteps, start_T)

        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts, clip_denoised=self.clip_denoised, **kwargs)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None, \
               verbose=True, timesteps=None, mask=None, x0=None, shape=None, **kwargs):
        if shape is None:
            shape = (batch_size, self.channels, self.temporal_length, *self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps,
                                  mask=mask, x0=x0, **kwargs)

    @torch.no_grad()
    def sample_log(self, cond, batch_size, ddim, ddim_steps, **kwargs):
        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.temporal_length, *self.image_size)
            samples, intermediates = ddim_sampler.sample(ddim_steps, batch_size, shape, cond, verbose=False, **kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size, return_intermediates=True, **kwargs)

        return samples, intermediates

    def configure_schedulers(self, optimizer):
        assert 'target' in self.scheduler_config
        scheduler_name = self.scheduler_config.target.split('.')[-1]
        interval = self.scheduler_config.interval
        frequency = self.scheduler_config.frequency
        if scheduler_name == "LambdaLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            scheduler.start_step = self.global_step
            lr_scheduler = {
                            'scheduler': LambdaLR(optimizer, lr_lambda=scheduler.schedule),
                            'interval': interval,
                            'frequency': frequency
            }
        elif scheduler_name == "CosineAnnealingLRScheduler":
            scheduler = instantiate_from_config(self.scheduler_config)
            decay_steps = scheduler.decay_steps
            last_step = -1 if self.global_step == 0 else scheduler.start_step
            lr_scheduler = {
                            'scheduler': CosineAnnealingLR(optimizer, T_max=decay_steps, last_epoch=last_step),
                            'interval': interval,
                            'frequency': frequency
            }
        else:
            raise NotImplementedError
        return lr_scheduler

class LatentVisualAudioDiffusion(LatentDiffusion):
    def __init__(self, img_cond_stage_config, image_proj_stage_config, audio_proj_stage_config=None, freeze_embedder=True, image_proj_model_trainable=True, select_keyframe=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_proj_model_trainable = image_proj_model_trainable
        self._init_embedder(img_cond_stage_config, freeze_embedder)
        self._init_img_ctx_projector(image_proj_stage_config, image_proj_model_trainable)
        
        if audio_proj_stage_config:
            self._init_audio_ctx_projector(audio_proj_stage_config)

        if self.audio_cond_key:
            # self.imagebind_model = imagebind_model.imagebind_huge(pretrained=True)
            if self.audio_proj_model.video_length == 12:
                audio_stride = 10
            elif self.audio_proj_model.video_length == 48:
                audio_stride = (10,4)
            elif self.audio_proj_model.video_length == 60:
                audio_stride = (10,3)
            else:
                raise ValueError("audio_stride not defined for video_length: ", audio_proj_model.video_length)
            print("audio_stride: ", audio_stride)
            self.imagebind_model = imagebind_model.ImageBindModel(
                                        vision_embed_dim=1280,
                                        vision_num_blocks=32,
                                        vision_num_heads=16,
                                        text_embed_dim=1024,
                                        text_num_blocks=24,
                                        text_num_heads=16,
                                        audio_stride=audio_stride,
                                        # audio_stride=(10, 4),
                                        audio_target_len=204,
                                        out_embed_dim=1024,
                                        audio_drop_path=0.1,
                                        imu_drop_path=0.7,

                                    )
            
            self.load_pretrained_imagebind_model()
            
            for param in self.imagebind_model.parameters():
                param.requires_grad = False
            # self.audio_adapter = nn.ModuleList([
            #         nn.Sequential(
            #             nn.LayerNorm(normalized_shape=768),
            #             nn.Linear(768, 768)
            #             )
            #     ])
        
        self.select_keyframe = select_keyframe
        # self.audio_adapter = nn.ModuleList([
        #         nn.Sequential(
        #             nn.LayerNorm(normalized_shape=768),
        #             nn.Linear(768, 1024)
        #             )
        #         for i in range(16)
        #     ])
    def load_pretrained_imagebind_model(self):
        if not os.path.exists(".checkpoints/imagebind_huge.pth"):
            print(
                "Downloading imagebind weights to .checkpoints/imagebind_huge.pth ..."
            )
            os.makedirs(".checkpoints", exist_ok=True)
            torch.hub.download_url_to_file(
                "https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth",
                ".checkpoints/imagebind_huge.pth",
                progress=True,
            )

        # model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))
        ckpt = torch.load(".checkpoints/imagebind_huge.pth")
        pos_embed = ckpt["modality_preprocessors.audio.pos_embedding_helper.pos_embed"]
        
        state_dict = self.imagebind_model.state_dict()
        model_pos_embed = state_dict["modality_preprocessors.audio.pos_embedding_helper.pos_embed"]

        if model_pos_embed.shape != pos_embed.shape:
            pos_embed = interpolate_pos_embed(pos_embed, model_pos_embed)
            ckpt["modality_preprocessors.audio.pos_embedding_helper.pos_embed"] = pos_embed

        unused = []
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
            else:
                unused.append(k)
        print(f"Unused keys: {unused}")
        self.imagebind_model.load_state_dict(state_dict)
    
    def _init_audio_ctx_projector(self, config):
        self.audio_proj_model = instantiate_from_config(config)

    def _init_img_ctx_projector(self, config, trainable):
        self.image_proj_model = instantiate_from_config(config)
        if not trainable:
            self.image_proj_model.eval()
            self.image_proj_model.train = disabled_train
            for param in self.image_proj_model.parameters():
                param.requires_grad = False

    def _init_embedder(self, config, freeze=True):
        self.embedder = instantiate_from_config(config)
        if freeze:
            self.embedder.eval()
            self.embedder.train = disabled_train
            for param in self.embedder.parameters():
                param.requires_grad = False
    
    def select_keyframes(self, img_emb, frame_idx, num_queries=16):
        b, l, c = img_emb.shape
        q = num_queries
        # print(img_emb.shape)
        img_emb = rearrange(img_emb, 'b (t q) c -> b t q c', q=q)

        frame_idx_expanded =  rearrange(frame_idx, 'b t -> b t 1 1').expand(-1, -1, q, c)

        img_emb = torch.take_along_dim(img_emb, frame_idx_expanded, dim=1)
        img_emb = rearrange(img_emb, 'b t q c -> b (t q) c')
    
        return img_emb

    def expand_keyframes(self, img_emb, target_len, num_queries=16):
        b, l, c = img_emb.shape
        q = num_queries
        # print(img_emb.shape)
        img_emb = rearrange(img_emb, 'b (t q) c -> b t q c', q=q)
        
        img_emb = rearrange(img_emb, 'b t q c -> b q c t')

        T = target_len
        if target_len == img_emb.shape[3]:
            print("Do not need to expand keyframes")
            img_emb = rearrange(img_emb, 'b q c t -> b (t q) c')
            return img_emb
    

        # inter_img_emb = torch.nn.functional.interpolate(
        #     img_emb, size=(c, T), mode="nearest"
        # )
        inter_img_emb = torch.nn.functional.interpolate(
            img_emb, size=(c, T), mode="bilinear"
        )
        inter_img_emb = rearrange(inter_img_emb, 'b q c t -> b (t q) c')

        # frame_idx_expanded =  rearrange(frame_idx, 'b t -> b t 1 1').expand(-1, -1, q, c)

        # img_emb = torch.take_along_dim(img_emb, frame_idx_expanded, dim=1)
        # img_emb = rearrange(img_emb, 'b t q c -> b (t q) c')
    
        return inter_img_emb

    # def get_audio_adpation(self, audio_input):
    #     # output shape torch.Size([2, 16, 19, 1024])
        
    #     output = []
    #     for i in range(audio_input.size(1)):
    #         output.append(self.audio_adapter[i](audio_input[:, i]))
    #     return torch.stack(output, dim=1)
         

    def shared_step(self, batch, random_uncond, **kwargs):
        # get input
        x, c, fs, frame_idx= self.get_batch_input(batch, random_uncond=random_uncond, return_fs=True, return_frame_idx=True)

        kwargs.update({"fs": fs.long()})
        kwargs.update({"frame_idx": frame_idx.long()})
        

        loss, loss_dict = self(x, c, **kwargs)
        return loss, loss_dict
    
    def get_batch_input(self, batch, random_uncond, return_first_stage_outputs=False, 
                        return_original_cond=False, return_fs=False, return_cond_frame=False, return_original_input=False,
                        return_audio_ori = False, return_frame_idx=False,
                        return_audio=False, **kwargs):

        ## x: b c t h w
        
        x = super().get_input(batch, self.first_stage_key) # torch.Size([B, 3, F, 320, 512])
        ## encode video frames x to z via a 2D encoder        
        z = self.encode_first_stage(x) # torch.Size([B, 4, F, 40, 64])
        num_frames = z.size(2)

        ## get caption condition
        cond_input = batch[self.cond_stage_key]

        if isinstance(cond_input, dict) or isinstance(cond_input, list):
            cond_emb = self.get_learned_conditioning(cond_input)
        else:
            cond_emb = self.get_learned_conditioning(cond_input.to(self.device))
        
        cond = {}
        ## to support classifier-free guidance, randomly drop out only text conditioning 5%, only image conditioning 5%, and both 5%.
        if random_uncond:
            random_num = torch.rand(x.size(0), device=x.device)
        else:
            random_num = torch.ones(x.size(0), device=x.device)  ## by doning so, we can get text embedding and complete img emb for inference
        prompt_mask = rearrange(random_num < 2 * self.uncond_prob, "n -> n 1 1")
        input_mask = 1 - rearrange((random_num >= self.uncond_prob).float() * (random_num < 3 * self.uncond_prob).float(), "n -> n 1 1 1")
        audio_mask = 1 - rearrange((random_num >= 2 * self.uncond_prob).float() * (random_num < 4 * self.uncond_prob).float(), "n -> n 1 1 1")
        
        null_prompt = self.get_learned_conditioning([""])
        prompt_imb = torch.where(prompt_mask, null_prompt, cond_emb.detach())

        ## get conditioning frame
        cond_frame_index = 0
        if self.rand_cond_frame:
            cond_frame_index = random.randint(0, self.model.diffusion_model.temporal_length-1)

        img = x[:,:,cond_frame_index,...]
        img = input_mask * img

        ## img: b c h w
        img_emb = self.embedder(img) ## b l c
        img_emb = self.image_proj_model(img_emb) # b, 960, 1024
        frame_idx = batch['frame_indices'].long()
        if self.interp_mode:
            is_keyframe = batch['is_keyframe']
        # key_frame_indices = batch['key_frame_indices']
        # import ipdb; ipdb.set_trace()
        
        
        if self.select_keyframe:
            img_emb = self.select_keyframes(img_emb, frame_idx, num_queries=self.image_proj_model.num_queries)

        if self.model.conditioning_key == 'hybrid':
            if self.interp_mode:
                ## starting frame + (L-2 empty frames) + ending frame
                img_cat_cond = torch.zeros_like(z)
                index_for_selection = torch.arange(0, z.size(2), 1, device=z.device)

                for j in range(z.size(0)):
                    key_frame_indices_ = torch.where(is_keyframe[j])[0]
                    img_cat_cond[j, :, key_frame_indices_, :, :] = z[j, :, key_frame_indices_, :, :]
                
                # Original
                # img_cat_cond[:,:,0,:,:] = z[:,:,0,:,:]
                # img_cat_cond[:,:,-1,:,:] = z[:,:,-1,:,:]
            else:
                ## simply repeat the cond_frame to match the seq_len of z
                img_cat_cond = z[:,:,cond_frame_index,:,:]
                img_cat_cond = img_cat_cond.unsqueeze(2)
                img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])

            cond["c_concat"] = [img_cat_cond] # b c t h w
        

        cond["c_crossattn"] = [torch.cat([prompt_imb, img_emb], dim=1)] ## concat in the seq_len dim

        # audio embedding
        # hard code
        # print(self.audio_cond_key)
        if self.audio_cond_key:
            audio = batch[self.audio_cond_key] # [B, 1, 128, 204])       
            audio = audio * audio_mask 
            c_audio = self.imagebind_model.forward_audio(audio) # 229, 768 -- 1 + 228
            

            # manually reshape, not used
            if not self.use_qformer:
                if self.select_keyframe:       
                    cond["c_audio"] = self.split_audio(c_audio, self.audio_proj_model.video_length)
                    cond["c_audio"] = rearrange(cond["c_audio"], 'b t l c -> b (t l) c')
                    cond["c_audio"] = self.select_keyframes(cond["c_audio"], frame_idx, num_queries=self.audio_proj_model.num_queries)
                    cond["c_audio"] = rearrange(cond["c_audio"], 'b (t l) c -> b t l c', l=self.audio_proj_model.num_queries) # 2*t, 8, 1024]
                else:
                    cond["c_audio"] = self.split_audio(c_audio, z.shape[2])
                # assert cond["c_audio"].shape[2] == 25
                # change openaimodel3d.py, line 410
                
            else:
                cond["c_audio"] = self.audio_proj_model(c_audio)
                if self.select_keyframe:
                    cond["c_audio"] = self.select_keyframes(cond["c_audio"], frame_idx, num_queries=self.audio_proj_model.num_queries)
                cond["c_audio"] = rearrange(cond["c_audio"], 'b (t l) c -> (b t) l c', l=self.audio_proj_model.num_queries) # 2*t, 8, 1024]
        else:
            c_audio = None
            cond["c_audio"] = None

        out = [z, cond]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([xrec])

        if return_original_cond:
            out.append(cond_input)
        if return_fs:
            if self.fps_condition_type == 'fs':
                fs = super().get_input(batch, 'frame_stride')
            elif self.fps_condition_type == 'fps':
                fs = super().get_input(batch, 'fps')
            elif self.fps_condition_type == 'kfs':
                fs = super().get_input(batch, 'frame_strides')
            elif self.fps_condition_type == 'kfs2fps':
                fs = super().get_input(batch, 'frame_strides_fps')
            elif self.fps_condition_type == 'kfidx':
                fs = super().get_input(batch, 'frame_indices').long()

                # print(super().get_input(batch, 'frame_strides'))
                # print(fs)
            out.append(fs)
        if return_cond_frame:
            assert cond_frame_index == 0
            out.append(x[:,:,cond_frame_index,...].unsqueeze(2))
        if return_original_input:
            out.append(x)
        if return_audio_ori:  
            # print(c_audio, self.audio_cond_key)  
            if c_audio is not None:
                out.append(batch['audio_waveform'][:, :, :batch['length']].mean(1))
                out.append(batch['sr'])
            else:
                out.append(None)
                out.append(None)
        if return_frame_idx:
            out.append(frame_idx)
        if return_audio:
            if c_audio is not None:
                out.append(audio)
            else:
                out.append(None)
        return out

    def _auto_split(self, n, n_chunk):
        '''
            automatically split into chunks with n_ele no differ by 1
            if n is not dividible by n_chunk, extra one's will be added to the heading chunks
        '''
        chunk_size = int(math.ceil(n / n_chunk))
        assert chunk_size >= 1, chunk_size
        
        chunk_start_indices = np.round(np.linspace(0, n-chunk_size, n_chunk, endpoint=True)).astype(np.int32)
        
        mask = torch.zeros(n_chunk, n).bool()
        for chunk_index, chunk_start_index in enumerate(chunk_start_indices):
            mask[chunk_index, chunk_start_index:chunk_start_index+chunk_size] = 1
        mask = mask.contiguous()
        assert mask.long().sum() == chunk_size * n_chunk, mask.long().sum()
        
        return mask

    def split_audio(self, audio_encodings, n_segment=16):
        video_length = n_segment
        batchsize = audio_encodings.size(0)
        T = (audio_encodings.size(1) - 1) // 12
        
        # audio = self.imagebind_model.forward_audio(audio) # 229, 768 -- 1 + 228
        n, t = 12, T # hard code
        
        segment_mask = self._auto_split(t, n_segment).unsqueeze(1).expand(n_segment, n, t).contiguous() # (s, n, t)
        segment_mask = rearrange(
            segment_mask, "s n t -> s (n t)"
        )
        segment_mask = torch.cat([
            torch.ones(n_segment, 1).bool(),
            segment_mask
        ], dim=1) # (s, 1+n*t) torch.Size([16, 229])
        segment_masks = repeat(segment_mask, "n s -> b n s", b=batchsize).contiguous().bool().to(self.device)

        audio_encodings  = repeat(audio_encodings, "b n c-> b f n c", f=video_length).contiguous().to(self.dtype)
        
        c_audio = torch.masked_select(audio_encodings, segment_masks.unsqueeze(-1)).view(batchsize, video_length, -1, 768)
        return c_audio


    @torch.no_grad()
    def log_images(self, batch, sample=True, ddim_steps=50, ddim_eta=1., plot_denoise_rows=False, \
                    unconditional_guidance_scale=1.0, mask=None, **kwargs):
        """ log images for LatentVisualDiffusion """
        ##### sampled_img_num: control sampled imgae for logging, larger value may cause OOM
        sampled_img_num = 1
        for key in batch.keys():
            batch[key] = batch[key][:sampled_img_num]

        ## TBD: currently, classifier_free_guidance sampling is only supported by DDIM
        use_ddim = ddim_steps is not None
        log = dict()
        print("running log_images")
        # ipdb> batch.keys() dict_keys(['video', 'caption', 'audio', 'path', 'fps', 'frame_stride'])
        z, c, xrec, xc, fs, cond_x, ori_video, ori_audio, sr, frame_idx, audio = self.get_batch_input(batch, random_uncond=False,
                                                return_first_stage_outputs=True,
                                                return_original_cond=True,
                                                return_fs=True,
                                                return_cond_frame=True,
                                                return_original_input=True,
                                                return_audio_ori=True,
                                                return_frame_idx=True,
                                                return_audio=True)
        # dict_keys(['c_concat', 'c_crossattn', 'c_audio'])
        # import ipdb; ipdb.set_trace()
        N = xrec.shape[0]
        log["image_condition"] = cond_x
        # log["reconst"] = xrec
        xc_with_fs = []
        for idx, content in enumerate(xc):
            if fs.dim() == 1:
                xc_with_fs.append(content + '_fs=' + str(fs[idx].item()) + '_frame_idx=' + str(frame_idx[idx, :].cpu().numpy()))
            else:
                xc_with_fs.append(content + '_fs=' + str(fs[idx, :].cpu().numpy()) + '_frame_idx=' + str(frame_idx[idx, :].cpu().numpy()))
            # xc_with_fs.append(content + '_fs=' + str(fs[idx,].item()))
        log["condition"] = xc_with_fs
        if ori_audio is not None:   
            log["audio"] = ori_audio
            log["sr"] = sr

        kwargs.update({"fs": fs.long()})
        kwargs.update({"frame_idx": frame_idx.long()})


        c_cat = None
        if sample:
            # get uncond embedding for classifier-free guidance sampling
            if unconditional_guidance_scale != 1.0:
                if isinstance(c, dict):
                    c_emb = c["c_crossattn"][0]
                    if 'c_concat' in c.keys():
                        c_cat = c["c_concat"][0]
                else:
                    c_emb = c

                if self.uncond_type == "empty_seq":
                    prompts = N * [""]
                    uc_prompt = self.get_learned_conditioning(prompts)
                elif self.uncond_type == "zero_embed":
                    uc_prompt = torch.zeros_like(c_emb)
                
                img = torch.zeros_like(xrec[:,:,0]) ## b c h w
                ## img: b c h w
                img_emb = self.embedder(img) ## b l c
                uc_img = self.image_proj_model(img_emb)
                if self.select_keyframe:
                    frame_idx = frame_idx.long()
                    uc_img = self.select_keyframes(uc_img, frame_idx, num_queries=self.image_proj_model.num_queries)

                uc = torch.cat([uc_prompt, uc_img], dim=1)
                
                # uncondition audio
                if self.audio_cond_key:
                    uc_audio = self.imagebind_model.forward_audio(torch.zeros_like(audio))

                    if not self.use_qformer:
                        if self.select_keyframe:       
                            uc_audio = self.split_audio(uc_audio, self.audio_proj_model.video_length)
                            uc_audio = rearrange(uc_audio, 'b t l c -> b (t l) c')
                            uc_audio = self.select_keyframes(uc_audio, frame_idx, num_queries=self.audio_proj_model.num_queries)
                            uc_audio = rearrange(uc_audio, 'b (t l) c -> b t l c', l=self.audio_proj_model.num_queries)
                        else:
                            uc_audio = self.split_audio(uc_audio, z.shape[2])
                        
                    else:
                        uc_audio = self.audio_proj_model(uc_audio)
                        if self.select_keyframe:
                            uc_audio = self.select_keyframes(uc_audio, frame_idx, num_queries=self.audio_proj_model.num_queries)
                        uc_audio = rearrange(uc_audio, 'b (t l) c -> (b t) l c', l=self.audio_proj_model.num_queries) # 2*t, 8, 1024]

                else:
                    uc_audio = None    

                
                ## hybrid case
                if isinstance(c, dict):
                    uc_hybrid = {"c_concat": [c_cat], "c_crossattn": [uc], "c_audio": uc_audio}
                    uc = uc_hybrid
            else:
                uc = None

            with self.ema_scope("Plotting"):
                samples, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta,
                                                         unconditional_guidance_scale=unconditional_guidance_scale,
                                                         unconditional_conditioning=uc, x0=z, **kwargs)
            x_samples = self.decode_first_stage(samples)
            # log["samples"] = x_samples
            log["samples_concat"] = torch.cat((x_samples, ori_video), dim=-1)
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid

        return log

    def configure_optimizers(self):
        """ configure_optimizers for LatentDiffusion """
        lr = self.learning_rate

        params = list(self.model.parameters())
        mainlogger.info(f"@Training [{len(params)}] Full Paramters.")

        if self.cond_stage_trainable:
            params_cond_stage = [p for p in self.cond_stage_model.parameters() if p.requires_grad == True]
            mainlogger.info(f"@Training [{len(params_cond_stage)}] Paramters for Cond_stage_model.")
            params.extend(params_cond_stage)
        
        if self.image_proj_model_trainable:
            mainlogger.info(f"@Training [{len(list(self.image_proj_model.parameters()))}] Paramters for Image_proj_model.")
            params.extend(list(self.image_proj_model.parameters()))   

        if self.learn_logvar:
            mainlogger.info('Diffusion model optimizing logvar')
            if isinstance(params[0], dict):
                params.append({"params": [self.logvar]})
            else:
                params.append(self.logvar)

        ## optimizer
        optimizer = torch.optim.AdamW(params, lr=lr)

        ## lr scheduler
        if self.use_scheduler:
            mainlogger.info("Setting up scheduler...")
            lr_scheduler = self.configure_schedulers(optimizer)
            return [optimizer], [lr_scheduler]
        
        return optimizer


class DiffusionWrapper(pl.LightningModule):
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None,
                c_adm=None, c_audio = None, s=None, mask=None, **kwargs):
        # temporal_context = fps is foNone
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, **kwargs)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, **kwargs)
        elif self.conditioning_key == 'hybrid':
            ## it is just right [b,c,t,h,w]: concatenate in channel dim
            ''' 
            x.shape torch.Size([2, 4, 16, 40, 64])
            c_concat[0].shape = torch.Size([2, 4, 16, 40, 64])
            c_crossattn[0].shape = torch.Size([2, 333, 1024])
            '''
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, c_audio = c_audio, **kwargs)
        elif self.conditioning_key == 'resblockcond':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        elif self.conditioning_key == 'hybrid-adm':
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, y=c_adm, **kwargs)
        elif self.conditioning_key == 'hybrid-time':
            assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s)
        elif self.conditioning_key == 'concat-time-mask':
            # assert s is not None
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t, context=None, s=s, mask=mask)
        elif self.conditioning_key == 'concat-adm-mask':
            # assert s is not None
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=None, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-adm-mask':
            cc = torch.cat(c_crossattn, 1)
            if c_concat is not None:
                xc = torch.cat([x] + c_concat, dim=1)
            else:
                xc = x
            out = self.diffusion_model(xc, t, context=cc, y=s, mask=mask)
        elif self.conditioning_key == 'hybrid-time-adm': # adm means y, e.g., class index
            # assert s is not None
            assert c_adm is not None
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc, s=s, y=c_adm)
        elif self.conditioning_key == 'crossattn-adm':
            assert c_adm is not None
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc, y=c_adm)
        else:
            raise NotImplementedError()

        return out
