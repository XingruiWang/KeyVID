import torchvision.io
from einops import rearrange, repeat
import numpy as np
import inspect
from typing import List, Optional, Union, Tuple
from tqdm import tqdm
import json
import math
import os
import PIL
import torch
from omegaconf import OmegaConf

from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.loaders import TextualInversionLoaderMixin
from diffusers.models import AutoencoderKL
from diffusers.schedulers import KarrasDiffusionSchedulers, PNDMScheduler
from diffusers.utils import logging
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.image_processor import VaeImageProcessor
import sys


import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from pytorch_lightning import seed_everything
from PIL import Image
sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
from lvdm.models.samplers.ddim import DDIMSampler
from lvdm.models.samplers.ddim_multiplecond import DDIMSampler as DDIMSampler_multicond
from utils.utils import instantiate_from_config
import random
from utils.utils import find_file_with_prefix
sys.path.append("../ASVA")
from avgen.models.unets import AudioUNet3DConditionModel
from avgen.models.audio_encoders import ImageBindSegmaskAudioEncoder
from avgen.data.utils import AudioMelspectrogramExtractor, get_evaluation_data, load_av_clips_uniformly, load_v_clips_uniformly, load_av_clips_keyframe, load_audio_clips_uniformly, load_image, waveform_to_melspectrogram
from avgen.utils import freeze_and_make_eval



from imagebind.data import waveform2melspec

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def get_views(video_length, window_size=16, stride=4):
    num_blocks_time = (video_length - window_size) // stride + 1
    views = []
    for i in range(num_blocks_time):
        t_start = int(i * stride)
        t_end = t_start + window_size
        views.append((t_start,t_end))
    return views
    
def waveform_to_melspectrogram(
        waveform,
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2.,
        mean=-4.268,
        std=9.138
):
    if isinstance(waveform, np.ndarray):
        waveform = torch.from_numpy(waveform)
    
    audio_length = waveform.shape[1]
    audio_target_length = int(clip_duration * sample_rate)
    
    audio_start_idx = 0
    if audio_length > audio_target_length:
        audio_start_idx = (audio_length - audio_target_length) // 2
    audio_end_idx = audio_start_idx + audio_target_length
    waveform_clip = waveform[:, audio_start_idx:audio_end_idx]
    
    waveform_melspec = waveform2melspec(
        waveform_clip, sample_rate, num_mel_bins, target_length
    )  # (1, n_mel, n_frame)
    # print(waveform_melspec.shape)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    audio_clip = normalize(waveform_melspec)
    
    return audio_clip  # (1, freq, time)


class AudioCondAnimationPipeline(DiffusionPipeline, TextualInversionLoaderMixin):
    """
    Pipeline for text-guided image to image generation using stable unCLIP.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        feature_extractor ([`CLIPImageProcessor`]):
            Feature extractor for image pre-processing before being encoded.
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`KarrasDiffusionSchedulers`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
    """
    text_encoder: CLIPTextModel
    tokenizer: CLIPTokenizer
    unet: AudioUNet3DConditionModel
    scheduler: KarrasDiffusionSchedulers
    vae: AutoencoderKL
    audio_encoder: ImageBindSegmaskAudioEncoder
    
    def __init__(
            self,
            text_encoder: CLIPTextModel,
            tokenizer: CLIPTokenizer,
            unet: AudioUNet3DConditionModel,
            scheduler: KarrasDiffusionSchedulers,
            vae: AutoencoderKL,
            audio_encoder: ImageBindSegmaskAudioEncoder,
            null_text_encodings_path: str = ""
    ):
        super().__init__()
        
        self.register_modules(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
            vae=vae,
            audio_encoder=audio_encoder
        )
        
        if null_text_encodings_path:
            self.null_text_encoding = torch.load(null_text_encodings_path).view(1, 77, 768)
        
        self.melspectrogram_shape = (128, 204)
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.audio_processor = AudioMelspectrogramExtractor()
    
    @torch.no_grad()
    def encode_text(
            self,
            texts,
            device,
            dtype,
            do_text_classifier_free_guidance,
            do_audio_classifier_free_guidance,
            text_encodings: Optional[torch.Tensor] = None
    ):
        if text_encodings is None:
            
            text_inputs = self.tokenizer(
                texts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            
            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            text_encodings = self.text_encoder(text_input_ids.to(device), attention_mask=attention_mask)
            text_encodings = text_encodings[0] # (b, n, d)
            
        else:
            if isinstance(text_encodings, (List, Tuple)):
                text_encodings = torch.cat(text_encodings)
        
        text_encodings = text_encodings.to(dtype=dtype, device=device)
        batch_size = len(text_encodings)
        
        # get unconditional embeddings for classifier free guidance
        if do_text_classifier_free_guidance:
            if not hasattr(self, "null_text_encoding"):
                uncond_token = ""
    
                max_length = text_encodings.shape[1]
                uncond_input = self.tokenizer(
                    uncond_token,
                    padding="max_length",
                    max_length=max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                
                if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                    attention_mask = uncond_input.attention_mask.to(device)
                else:
                    attention_mask = None
                
                uncond_text_encodings = self.text_encoder(
                    uncond_input.input_ids.to(device),
                    attention_mask=attention_mask,
                )
                uncond_text_encodings = uncond_text_encodings[0]
                
            else:
                uncond_text_encodings = self.null_text_encoding
            
            uncond_text_encodings = repeat(uncond_text_encodings, "1 n d -> b n d", b=batch_size).contiguous()
            uncond_text_encodings = uncond_text_encodings.to(dtype=dtype, device=device)
        
        if do_text_classifier_free_guidance and do_audio_classifier_free_guidance: # dual cfg
            text_encodings = torch.cat([uncond_text_encodings, text_encodings, text_encodings])
        elif do_text_classifier_free_guidance: # only text cfg
            text_encodings = torch.cat([uncond_text_encodings, text_encodings])
        elif do_audio_classifier_free_guidance: # only audio cfg
            text_encodings = torch.cat([text_encodings, text_encodings])
        
        return text_encodings
    
    @torch.no_grad()
    def encode_audio(
            self,
            audios: Union[List[np.ndarray], List[torch.Tensor]],
            video_length: int = 12,
            do_text_classifier_free_guidance: bool = False,
            do_audio_classifier_free_guidance: bool = False,
            device: torch.device = torch.device("cuda:0"),
            dtype: torch.dtype = torch.float32
    ):
        batch_size = len(audios)
        melspectrograms = self.audio_processor(audios).to(device=device, dtype=dtype) # (b c n t)
        
        # audio_encodings: (b, n, c)
        # audio_masks: (b, s, n)
        _, audio_encodings, audio_masks = self.audio_encoder(
            melspectrograms, normalize=False, return_dict=False
        )
        audio_encodings = repeat(audio_encodings, "b n c -> b f n c", f=video_length)
        
        if do_audio_classifier_free_guidance:
            null_melspectrograms = torch.zeros(1, 1, *self.melspectrogram_shape).to(device=device, dtype=dtype)
            _, null_audio_encodings, null_audio_masks = self.audio_encoder(
                null_melspectrograms, normalize=False, return_dict=False
            )
            null_audio_encodings = repeat(null_audio_encodings, "1 n c -> b f n c", b=batch_size, f=video_length)
        
        if do_text_classifier_free_guidance and do_audio_classifier_free_guidance: # dual cfg
            audio_encodings = torch.cat([null_audio_encodings, null_audio_encodings, audio_encodings])
            audio_masks = torch.cat([null_audio_masks, null_audio_masks, audio_masks])
        elif do_text_classifier_free_guidance: # only text cfg
            audio_encodings = torch.cat([audio_encodings, audio_encodings])
            audio_masks = torch.cat([audio_masks, audio_masks])
        elif do_audio_classifier_free_guidance: # only audio cfg
            audio_encodings = torch.cat([null_audio_encodings, audio_encodings])
            audio_masks = torch.cat([null_audio_masks, audio_masks])
        
        return audio_encodings, audio_masks
    
    @torch.no_grad()
    def encode_latents(self, image: torch.Tensor):
        dtype = self.vae.dtype
        image = image.to(device=self.device, dtype=dtype)
        image_latents = self.vae.encode(image).latent_dist.sample() * self.vae.config.scaling_factor
        return image_latents
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.decode_latents
    @torch.no_grad()
    def decode_latents(self, latents):
        dtype = next(self.vae.parameters()).dtype
        latents = latents.to(dtype=dtype)
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1).cpu().float()  # ((b t) c h w)
        return image
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta
        
        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_video_latents(
            self,
            image_latents: torch.Tensor,
            num_channels_latents: int,
            video_length: int = 12,
            height: int = 256,
            width: int = 256,
            device: torch.device = torch.device("cuda"),
            dtype: torch.dtype = torch.float32,
            generator: Optional[torch.Generator] = None,
    ):
        batch_size = len(image_latents)
        shape = (
            batch_size,
            num_channels_latents,
            video_length-1,
            height // self.vae_scale_factor,
            width // self.vae_scale_factor
        )
        
        image_latents = image_latents.unsqueeze(2) # (b c 1 h w)
        rand_noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        noise_latents = torch.cat([image_latents, rand_noise], dim=2)
        
        # scale the initial noise by the standard deviation required by the scheduler
        noise_latents = noise_latents * self.scheduler.init_noise_sigma
        
        return noise_latents
    
    @torch.no_grad()
    def __call__(
            self,
            images: List[PIL.Image.Image],
            audios: Union[List[np.ndarray], List[torch.Tensor]],
            texts: List[str],
            text_encodings: Optional[List[torch.Tensor]] = None,
            video_length: int = 12,
            height: int = 256,
            width: int = 256,
            num_inference_steps: int = 20,
            audio_guidance_scale: float = 4.0,
            text_guidance_scale: float = 1.0,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True
    ):
        # 0. Default height and width to unet
        device = self.device
        dtype = self.dtype
        
        batch_size = len(images)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        
        do_text_classifier_free_guidance = (text_guidance_scale > 1.0)
        do_audio_classifier_free_guidance = (audio_guidance_scale > 1.0)
        
        # 1. Encoder text into ((k b) f n d)
        text_encodings = self.encode_text(
            texts=texts,
            text_encodings=text_encodings,
            device=device,
            dtype=dtype,
            do_text_classifier_free_guidance=do_text_classifier_free_guidance,
            do_audio_classifier_free_guidance=do_audio_classifier_free_guidance
        ) # ((k b), n, d)
        text_encodings = repeat(text_encodings, "b n d -> b t n d", t=video_length).to(device=device, dtype=dtype)
        
        # 2. Encode audio
        # audio_encodings: ((k b), n, d)
        # audio_masks: ((k b), s, n)
        audio_encodings, audio_masks = self.encode_audio(
            audios, video_length, do_text_classifier_free_guidance, do_audio_classifier_free_guidance, device, dtype
        )
        
        # 3. Prepare image latent
        image = self.image_processor.preprocess(images)
        image_latents = self.encode_latents(image).to(device=device, dtype=dtype)  # (b c h w)
        
        # 4. Prepare unet noising video latents
        video_latents = self.prepare_video_latents(
            image_latents=image_latents,
            num_channels_latents=self.unet.config.in_channels,
            video_length=video_length,
            height=height,
            width=width,
            dtype=dtype,
            device=device,
            generator=generator,
        )  # (b c f h w)
        
        # 5. Prepare timesteps and extra step kwargs
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta=0.0)
        
        # 7. Denoising loop
        for i, t in enumerate(self.progress_bar(timesteps)):
            latent_model_input = [video_latents]
            if do_text_classifier_free_guidance:
                latent_model_input.append(video_latents)
            if do_audio_classifier_free_guidance:
                latent_model_input.append(video_latents)
            latent_model_input = torch.cat(latent_model_input)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_encodings,
                audio_encoder_hidden_states=audio_encodings,
                audio_attention_mask=audio_masks
            ).sample
            
            # perform guidance
            if do_text_classifier_free_guidance and do_audio_classifier_free_guidance: # dual cfg
                noise_pred_uncond, noise_pred_text, noise_pred_text_audio = noise_pred.chunk(3)
                noise_pred = noise_pred_uncond + \
                             text_guidance_scale * (noise_pred_text - noise_pred_uncond) + \
                             audio_guidance_scale * (noise_pred_text_audio - noise_pred_text)
            elif do_text_classifier_free_guidance: # only text cfg
                noise_pred_audio, noise_pred_text_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_audio + \
                            text_guidance_scale * (noise_pred_text_audio - noise_pred_audio)
            elif do_audio_classifier_free_guidance: # only audio cfg
                noise_pred_text, noise_pred_text_audio = noise_pred.chunk(2)
                noise_pred = noise_pred_text + \
                             audio_guidance_scale * (noise_pred_text_audio - noise_pred_text)
            
            # First frame latent will always server as unchanged condition
            video_latents[:, :, 1:, :, :] = self.scheduler.step(noise_pred[:, :, 1:, :, :], t, video_latents[:, :, 1:, :, :], **extra_step_kwargs).prev_sample
            video_latents = video_latents.contiguous()
        
        # 8. Post-processing
        video_latents = rearrange(video_latents, "b c f h w -> (b f) c h w")
        videos = self.decode_latents(video_latents).detach().cpu()
        videos = rearrange(videos, "(b f) c h w -> b f c h w", f=video_length) # value range [0, 1]
        
        if not return_dict:
            return videos
        
        return {"videos": videos}


@torch.no_grad()
def generate_videos(
        pipeline,
        image_path: str = '',
        audio_path: str = '',
        video_path: str = '',
        category: str = '',
        category_text_encoding: Optional[torch.Tensor] = None,
        image_size: Tuple[int, int] = (256, 256),
        video_fps: int = 6,
        video_num_frame: int = 12,
        num_clips_per_video: int = 3,
        audio_guidance_scale: float = 4.0,
        text_guidance_scale: float = 1.0,
        seed: int = 0,
        save_template: str = "",
        device: torch.device = torch.device("cuda"),
):
    # Prioritize loading from image_path and audio_path for image and audio, respectively
    # Otherwise, load from video_path
    # Can not specify all three
    assert not (image_path and audio_path and video_path), "Can not specify image_path, audio_path, video_path all three"
    clip_duration = video_num_frame/video_fps
    
    images = None
    audios = None
    
    if image_path:
        image = load_image(image_path, image_size)
        images = [image,] * num_clips_per_video
    
    if audio_path:
        audios = load_audio_clips_uniformly(audio_path, clip_duration, num_clips_per_video, load_audio_as_melspectrogram=False)
    
    if video_path:
        load_videos, load_audios = load_av_clips_uniformly(
            video_path, video_fps, video_num_frame, image_size, num_clips_per_video,
            load_audio_as_melspectrogram=False
        )
        
        if images is None:
            images = [video[0] for video in load_videos]
        if audios is None:
            audios = load_audios
    
    # convert images to PIL Images
    images = [
        PIL.Image.fromarray((255 * image).byte().permute(1, 2, 0).contiguous().numpy()) for image in images
    ]
    
    generated_video_list = []
    generated_audio_list = []
    
    generator = torch.Generator(device=device)
    for k, (image, audio) in enumerate(zip(images, audios)):
        generator.manual_seed(seed)
        generated_video = pipeline(
            images=[image],
            audios=[audio],
            texts=[category],
            text_encodings=[category_text_encoding] if category_text_encoding is not None else None,
            video_length=video_num_frame,
            height=image_size[0],
            width=image_size[1],
            num_inference_steps=50,
            audio_guidance_scale=audio_guidance_scale,
            text_guidance_scale=text_guidance_scale,
            generator=generator,
            return_dict=False
        )[0]  # (f c h w) in range [0, 1]
        generated_video = (generated_video.permute(0, 2, 3, 1).contiguous() * 255).byte()
        
        if save_template:
            save_path = f"{save_template}_clip-{k:02d}.mp4"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.io.write_video(
                filename=save_path,
                video_array=generated_video,
                fps=video_fps,
                audio_array=audio,
                audio_fps=16000,
                audio_codec="aac"
            )
        else:
            generated_video_list.append(generated_video)
            generated_audio_list.append(audio)
    
    if save_template:
        return
        
    return generated_video_list, generated_audio_list


@torch.no_grad()
def generate_videos_for_dataset(
        exp_root: str,
        checkpoint: int,
        dataset: str = "AVSync15",
        image_size: Tuple[int, int] = (256, 256),
        video_fps: int = 6,
        video_num_frame: int = 12,
        num_clips_per_video: int = 3,
        audio_guidance_scale: float = 4.0,
        text_guidance_scale: float = 1.0,
        random_seed: int = 0,
        device: torch.device = torch.device(f"cuda"),
        dtype: torch.dtype = torch.float16
):
    
    checkpoint_path = f"{exp_root}/ckpts/checkpoint-{checkpoint}/modules"
    save_root = f"{exp_root}/evaluations/checkpoint-{checkpoint}/AG-{audio_guidance_scale}_TG-{text_guidance_scale}/seed-{random_seed}/videos"
    
    # 1. Prepare datasets and precomputed features
    video_root, filenames, categories, video_type = get_evaluation_data(dataset)
    
    null_text_encoding_path = "./pretrained/openai-clip-l_null_text_encoding.pt"
    if dataset == "TheGreatestHits":
        category_text_encoding = torch.load("./datasets/TheGreatestHits/class_clip_text_encodings_stable-diffusion-v1-5.pt", map_location="cpu")
        category_mapping = {"hitting with a stick": "hitting with a stick"}
        category_text_encoding_mapping = {"hitting with a stick": category_text_encoding}
    elif dataset == "Landscapes":
        category_mapping = json.load(open('./datasets/Landscapes/class_mapping.json', 'r'))
        category_text_encoding_mapping = torch.load('./datasets/Landscapes/class_clip_text_encodings_stable-diffusion-v1-5.pt', map_location="cpu")
    elif dataset == "AVSync15":
        category_mapping = json.load(open('./datasets/AVSync15/class_mapping.json', 'r'))
        category_text_encoding_mapping = torch.load('./datasets/AVSync15/class_clip_text_encodings_stable-diffusion-v1-5.pt', map_location="cpu")
    else:
        raise Exception()
    
    # 2. Prepare models
    pretrained_stable_diffusion_path = "./pretrained/stable-diffusion-v1-5"
    
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_stable_diffusion_path, subfolder="tokenizer")
    scheduler = PNDMScheduler.from_pretrained(pretrained_stable_diffusion_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_stable_diffusion_path, subfolder="text_encoder").to(device=device, dtype=dtype)
    vae = AutoencoderKL.from_pretrained(pretrained_stable_diffusion_path, subfolder="vae").to(device=device, dtype=dtype)
    audio_encoder = ImageBindSegmaskAudioEncoder(n_segment=video_num_frame).to(device=device, dtype=dtype)
    freeze_and_make_eval(audio_encoder)
    unet = AudioUNet3DConditionModel.from_pretrained(checkpoint_path, subfolder="unet").to(device=device, dtype=dtype)

    pipeline = AudioCondAnimationPipeline(
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        vae=vae,
        audio_encoder=audio_encoder,
        null_text_encodings_path=null_text_encoding_path
    )
    pipeline.to(torch_device=device, dtype=dtype)
    pipeline.set_progress_bar_config(disable=True)
    
    # 3. Generating one by one

    for filename, category in tqdm(zip(filenames, categories), total=len(filenames)):
        video_path = os.path.join(video_root, filename)
        save_template = os.path.join(save_root, filename.replace(".mp4", ""))
        
        category_text_encoding = category_text_encoding_mapping[category_mapping[category]].view(1, 77, 768)
        
        generate_videos(
            pipeline,
            video_path=video_path,
            category_text_encoding=category_text_encoding,
            image_size=image_size,
            video_fps=video_fps,
            video_num_frame=video_num_frame,
            num_clips_per_video=num_clips_per_video,
            text_guidance_scale=text_guidance_scale,
            audio_guidance_scale=audio_guidance_scale,
            seed=random_seed,
            save_template=save_template,
            device=device
        )


def load_data(video_path, filename, video_size=(256,256), video_frames=16, video_fps=8, num_clips_per_video=3):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    ## load prompts
    category = filename.split("/")[0]


    ## load video
    
    load_videos, load_audios = load_av_clips_uniformly(
        video_path, video_fps, video_frames, video_size, num_clips_per_video,
        load_audio_as_melspectrogram=True
    )
    
    # import ipdb; ipdb.set_trace()
    
    
    images = [video[0] for video in load_videos]
    audios = load_audios
    
    # convert images to PIL Images
    images = [
        PIL.Image.fromarray((255 * image).byte().permute(1, 2, 0).contiguous().numpy()) for image in images
    ]

    filename_list = []
    caption_list = []
    data_list = []
    audio_list = []

    n_samples = len(images)
    for idx in range(n_samples):
        
        # image = PIL.Image.fromarray(images[idx])
        # image = Image.open(file_list[idx]).convert('RGB')
        image_tensor = transform(images[idx]).unsqueeze(1) # [c,1,h,w]
        
        frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
        sub_filename = filename

        caption = " ".join(category.split("_"))

        # if interp:
        #     image1 = Image.open(file_list[2*idx]).convert('RGB')
        #     image_tensor1 = transform(image1).unsqueeze(1) # [c,1,h,w]
        #     image2 = Image.open(file_list[2*idx+1]).convert('RGB')
        #     image_tensor2 = transform(image2).unsqueeze(1) # [c,1,h,w]
        #     frame_tensor1 = repeat(image_tensor1, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
        #     frame_tensor2 = repeat(image_tensor2, 'c t h w -> c (repeat t) h w', repeat=video_frames//2)
        #     frame_tensor = torch.cat([frame_tensor1, frame_tensor2], dim=1)
        #     _, filename = os.path.split(file_list[idx*2])
        # else:
        #     image = Image.open(file_list[idx]).convert('RGB')
        #     image_tensor = transform(image).unsqueeze(1) # [c,1,h,w]
        #     frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
        #     _, filename = os.path.split(file_list[idx])

        data_list.append(frame_tensor.unsqueeze(0))
        caption_list.append(caption)
        filename_list.append(sub_filename)
        audio_list.append(audios[idx].unsqueeze(0))

        
    return filename_list, data_list, caption_list, audio_list



def load_data_batch(data_dir, filenames, keyframe_gen_dir, video_size=(256,256), video_frames=16, video_fps=8, num_clips_per_video=3, fps_condition_type="kfs", swap_audio=False, batch_filename_neg=[]):
    transform = transforms.Compose([
        transforms.Resize(min(video_size)),
        transforms.CenterCrop(video_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    
    ## load prompts
    filename_list = []
    caption_list = []
    data_list = []
    audio_list = []
    raw_audios=[]
    
    if not swap_audio:
        filenames_2 = filenames
    else:
        filenames_2 = batch_filename_neg
    
    for filename, filename_2 in zip(filenames, filenames_2):
        video_path = os.path.join(data_dir, filename)
        category = filename.split("/")[0]

        # load keyframe idx
        
        frame_npy_name = video_path.split("/")[-1].split(".")[0]+".npy"

        # this is ground truth keyframe idx
        # frame_npy = np.load(find_file_with_prefix(os.path.join('/dockerx/groups/data/AVSync15/test_curves_npy', category), frame_npy_name[:11]))
        
        # this is predicted keyframe idx
        # keyframe_save_path = '/dockerx/share/Dynamicrafter_audio/save/keyframe_idx'
        keyframe_save_path = '/dockerx/groups/keyframe_idx'

        keyframes = np.load(os.path.join(keyframe_save_path, category, frame_npy_name))
        keyframes = torch.tensor(keyframes).to(torch.float64)

        # # # ================= load uniform
        # keyframes = np.linspace(0, 48, num=13, dtype=int)[:-1]
        # # expand / repead
        # keyframes = np.resize(keyframes, (3, 12))
        # keyframes = torch.tensor(keyframes).to(torch.float64)
        # ====================

        # keyframe_gen_dir = "/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_audio_7.5_img_2.0/samples"
        # keyframe_gen_dir = "/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_kf_add_idx_add_fps/open_domain-kf_audio_7.5_img_2.0/samples/"
        
        # keyframe_gen_dir = "/dockerx/share/Dynamicrafter_audio/save/asva/epoch=849-step=10200-kf_audio_7.5_img_2.0/samples"
        # keyframe_gen_dir = "/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_uniform/epoch=549-step=6600_audio_4.0_img_2.0_inpainting_step_0/samples"

        # keyframe_gen_dir = "/dockerx/groups/tmp/asva_12_kf_no_idx/epoch=849-step=10200-kf_audio_7.5_img_2.0/samples"

        
        keyframe_gen_path = [ os.path.join(keyframe_gen_dir, filename.replace('.mp4', f'_clip-0{str(i)}.mp4')) for i in range(num_clips_per_video)]
        keyframe_clips = []
        for keyframe_path in keyframe_gen_path:
            keyframe_clip, waveform, info = torchvision.io.read_video(keyframe_path, pts_unit="sec") # [T, 720, 1280, 3])
            keyframe_clip = (keyframe_clip / 255 - 0.5) * 2
            keyframe_clip = keyframe_clip.permute(3, 0, 1, 2) # [T, 3, 720, 1280]
            keyframe_clips.append(keyframe_clip.unsqueeze(0).cuda())
        ## load video
        
        load_videos, load_audios = load_av_clips_uniformly(
            video_path, video_fps, video_frames, video_size, num_clips_per_video,
            load_audio_as_melspectrogram=False
        )

        # load_videos, _ = load_v_clips_uniformly(
        #     "/dockerx/local/AVSync15/open_video/20250305_1237_Hammer_Strikes_Wood_simple_compose_01jnkp3kzbfhr8kngd25an5r0e.mp4", video_fps, video_frames, video_size, num_clips_per_video,
        #     load_audio_as_melspectrogram=True
        # )
        
        full_frame_idx_array = np.linspace(0, 48, num=video_frames+1, dtype=int)[:-1]
        
        if fps_condition_type == "kfs":
            frame_strides_array = full_frame_idx_array[1:]-full_frame_idx_array[0:-1]
            frame_strides_array = np.insert(frame_strides_array, 0, frame_strides_array[0])
        elif fps_condition_type == "kfidx":
            frame_strides_array = full_frame_idx_array
        elif fps_condition_type =="kfs2fps" or fps_condition_type == 'fps':
            frame_strides_array = np.array([24]*48)
        
        frame_strides_array = full_frame_idx_array

        
        frame_strides = torch.tensor(frame_strides_array).unsqueeze(0).repeat(num_clips_per_video, 1).to(torch.float64)

        images = [video[0] for video in load_videos]
        raw_audios = load_audios
        
        audios = [waveform_to_melspectrogram(audio) for audio in raw_audios]
        
        # convert images to PIL Images
        images = [
            PIL.Image.fromarray((255 * image).byte().permute(1, 2, 0).contiguous().numpy()) for image in images
        ]

        n_samples = len(images)
        for idx in range(n_samples):
            
            # image = PIL.Image.fromarray(images[idx])
            # image = Image.open(file_list[idx]).convert('RGB')
            image_tensor = transform(images[idx]).unsqueeze(1) # [c,1,h,w]
            
            frame_tensor = repeat(image_tensor, 'c t h w -> c (repeat t) h w', repeat=video_frames)
            sub_filename = filename

            caption = " ".join(category.split("_"))

            data_list.append(frame_tensor.unsqueeze(0))
            caption_list.append(caption)
            filename_list.append(sub_filename)
            audio_list.append(audios[idx].unsqueeze(0))

        
    return filename_list, data_list, caption_list, audio_list, raw_audios, load_videos,  keyframes, frame_strides, keyframe_clips


def get_latent_z(model, videos):
    b, c, t, h, w = videos.shape
    x = rearrange(videos, 'b c t h w -> (b t) c h w')
    z = model.encode_first_stage(x)
    z = rearrange(z, '(b t) c h w -> b c t h w', b=b, t=t)
    return z



def image_guided_synthesis(model, prompts, videos, audios, keyframe_clips, keyframes, frame_strides, \
                        noise_shape, n_samples=1, ddim_steps=50, ddim_eta=1., inpainting_end_step = 0, \
                        unconditional_guidance_scale=1.0, cfg_img=None, cfg_audio=None, fs=None, text_input=False, multiple_cond_cfg=False, loop=False, interp=False, timestep_spacing='uniform', guidance_rescale=0.0, **kwargs):
    ddim_sampler = DDIMSampler(model) if not multiple_cond_cfg else DDIMSampler_multicond(model)
    batch_size = noise_shape[0]
    fs = torch.tensor([fs] * batch_size, dtype=torch.long, device=model.device)

    if not text_input:
        prompts = [""]*batch_size
    
    img = videos[:,:,0] # bchw
    img_emb = model.embedder(img) ## blc
    img_emb = model.image_proj_model(img_emb) # torch.Size([1, 256, 1024])    
    frame_idx = frame_strides.to(img_emb).long()

    # no need to select keyframes, as we are expanding
    # if model.select_keyframe:
        # img_emb = model.select_keyframes(img_emb, frame_idx, num_queries=model.image_proj_model.num_queries)

    img_emb = model.expand_keyframes(img_emb, 48, num_queries=model.image_proj_model.num_queries)
    
    cond_emb = model.get_learned_conditioning(prompts)
    cond = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]} # torch.Size([1, 333, 1024]) = torch.Size([1, 77, 1024]) +  torch.Size([1, 256, 1024])

    c_audio = model.imagebind_model.forward_audio(audios) 
    # c_audio = model.imagebind_model.forward_audio(torch.zeros_like(audios)) # zero condition debug


    if model.select_keyframe:       
        cond["c_audio"] = model.split_audio(c_audio, model.audio_proj_model.video_length)
        cond["c_audio"] = rearrange(cond["c_audio"], 'b t l c -> b (t l) c')
        cond["c_audio"] = model.select_keyframes(cond["c_audio"], frame_idx, num_queries=model.audio_proj_model.num_queries)
        cond["c_audio"] = rearrange(cond["c_audio"], 'b (t l) c -> b t l c', l=model.audio_proj_model.num_queries) # 2*t, 8, 1024]
    else:
        cond["c_audio"] = model.split_audio(c_audio, model.audio_proj_model.video_length)


    cond["c_audio"] = rearrange(cond["c_audio"], 'b t l c -> b (t l) c')
    cond["c_audio"] = model.expand_keyframes(cond["c_audio"], 48, num_queries=model.audio_proj_model.num_queries)
    cond["c_audio"] = rearrange(cond["c_audio"], 'b (t l) c -> b t l c', l=model.audio_proj_model.num_queries) # 2*t, 8, 1024] 


    if model.model.conditioning_key == 'hybrid':
        z = get_latent_z(model, videos) # b c t h w
        z_keyframe = get_latent_z(model, keyframe_clips) # b c t h w

        if interp: # change here

            img_cat_cond = torch.zeros_like(z)

            for j in range(z.size(0)):
                key_frame_indices_ = keyframes[j].long().tolist()
                img_cat_cond[j, :, key_frame_indices_, :, :] = z_keyframe[j, :, :, :, :]

        else:
            img_cat_cond = z[:,:,:1,:,:]
            img_cat_cond = repeat(img_cat_cond, 'b c t h w -> b c (repeat t) h w', repeat=z.shape[2])
        cond["c_concat"] = [img_cat_cond] # b c t h w
    # import ipdb; ipdb.set_trace()
    
    if unconditional_guidance_scale != 1.0 or multiple_cond_cfg:
        if model.uncond_type == "empty_seq":
            prompts = batch_size * [""]
            uc_emb = model.get_learned_conditioning(prompts)
        elif model.uncond_type == "zero_embed":
            uc_emb = torch.zeros_like(cond_emb)
        uc_img_emb = model.embedder(torch.zeros_like(img)) ## b l c
        uc_img_emb = model.image_proj_model(uc_img_emb)
        if model.select_keyframe:
            uc_img_emb = model.select_keyframes(uc_img_emb, keyframes.to(uc_img_emb).long(), num_queries=model.image_proj_model.num_queries)

        uc_img_emb = model.expand_keyframes(uc_img_emb, 48, num_queries=model.image_proj_model.num_queries)

        uc_audio = model.imagebind_model.forward_audio(torch.zeros_like(audios))
        
        if model.select_keyframe:       
            uc_audio = model.split_audio(uc_audio, model.audio_proj_model.video_length)
            uc_audio = rearrange(uc_audio, 'b t l c -> b (t l) c')
            uc_audio = model.select_keyframes(uc_audio, frame_idx, num_queries=model.audio_proj_model.num_queries)
            uc_audio = rearrange(uc_audio, 'b (t l) c -> b t l c', l=model.audio_proj_model.num_queries) # 2*t, 8, 1024]
        else:
            uc_audio = model.split_audio(uc_audio, z.shape[2])

        uc_audio = rearrange(uc_audio, 'b t l c -> b (t l) c')
        uc_audio = model.expand_keyframes(uc_audio, 48, num_queries=model.audio_proj_model.num_queries)
        uc_audio = rearrange(uc_audio, 'b (t l) c -> b t l c', l=model.audio_proj_model.num_queries) # 2*t, 8, 1024]
        
        uc = {"c_crossattn": [torch.cat([uc_emb,uc_img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc["c_concat"] = [img_cat_cond]
        uc["c_audio"] = uc_audio
    else:
        uc = None

    ## we need one more unconditioning image=yes, text=""
    if multiple_cond_cfg and cfg_img != 1.0:
        uc_2 = {"c_crossattn": [torch.cat([uc_emb,img_emb],dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_2["c_concat"] = [img_cat_cond]
        uc_2["c_audio"] = uc_audio
        kwargs.update({"unconditional_conditioning_img_nonetext": uc_2})
    else:
        kwargs.update({"unconditional_conditioning_img_nonetext": None})

    if multiple_cond_cfg and cfg_audio != 1.0:
        uc_3 = {"c_crossattn": [torch.cat([cond_emb,img_emb], dim=1)]}
        if model.model.conditioning_key == 'hybrid':
            uc_3["c_concat"] = [img_cat_cond]
        uc_3["c_audio"] = uc_audio
        kwargs.update({"unconditional_conditioning_img_text_noneaudio": uc_3})
    else:
        kwargs.update({"unconditional_conditioning_img_text_noneaudio": None})


    z0 = None
    cond_mask = None

    # --fps 16 \
    # --frames 64 \
    # --window_size 16 \
    # --window_stride 4 

    views = get_views(48, 12, 6)

    batch_variants = []
    
    # x_T_total = torch.randn([n_samples, 1, channels, 48, h, w], device=model.device).repeat(1, batch_size, 1, 1, 1, 1)
    # mark
    # import ipdb; ipdb.set_trace()
    
    # print(keyframes)
    keyframe_for_x_T = keyframes[0].long().tolist() + [47]
    x_T_total = torch.randn(noise_shape, device=model.device)[None, ...]
    x_T_keyframe = x_T_total[:, :, :,keyframe_for_x_T]

    x_left = keyframe_for_x_T[0]

    for i in range(1, len(keyframe_for_x_T)):
        x_right = keyframe_for_x_T[i]
        
        length = x_right - x_left
        for j in range(x_left+1, x_right):
            x_T_total[:, :, :, j] = (j-x_left) / length * x_T_total[:, :, :, x_left] + (x_right - j) / length * x_T_total[:, :, :, x_right]

        x_left = x_right


    # for frame_index in range(12, 48, 6):
    #     list_index = list(range(frame_index-12, frame_index+6-12))
    #     random.shuffle(list_index)
    #     x_T_total[:, :, :, frame_index:frame_index+4] = x_T_total[:, :, :, list_index]

    for _ in range(n_samples):
        if z0 is not None:
            cond_z0 = z0.clone()
            kwargs.update({"clean_cond": True})
        else:
            cond_z0 = None
        if ddim_sampler is not None:

            
            samples, _ = ddim_sampler.sample(S=ddim_steps,
                                            conditioning=cond,
                                            batch_size=batch_size,
                                            shape=noise_shape[1:],
                                            verbose=False,
                                            unconditional_guidance_scale=unconditional_guidance_scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            cfg_img=cfg_img, 
                                            cfg_audio=cfg_audio, 
                                            mask=cond_mask,
                                            x0=cond_z0,
                                            x_keyframe = z_keyframe, 
                                            keyframe_idx = keyframes,
                                            end_step = inpainting_end_step,
                                            x_T = x_T_total[_],
                                            fs=fs,
                                            frame_idx=frame_idx,
                                            timestep_spacing=timestep_spacing,
                                            guidance_rescale=guidance_rescale,
                                            context_keyframes = keyframes,
                                            context_next=views,
                                            **kwargs
                                            )
            
        ## reconstruct from latent to pixel space
        batch_images = model.decode_first_stage(samples)
        batch_variants.append(batch_images)
    ## variants, batch, c, t, h, w
    # import ipdb; ipdb.set_trace()
    
    batch_variants = torch.stack(batch_variants)
    return batch_variants.permute(1, 0, 2, 3, 4, 5)


def load_model_checkpoint(model, ckpt):
    state_dict = torch.load(ckpt, map_location="cpu")
    state_dict = {k:v for k,v in state_dict["state_dict"].items() if "imagebind_model" not in k}

    model.load_state_dict(state_dict, strict=False)

    model.load_pretrained_imagebind_model()

    return model


def check_exist(samples, audios, filenames, fakedir, fps=12, loop=False, idx = [0]):
    # import ipdb; ipdb.set_trace()
    
    for i, filename in enumerate(filenames):
        for j in range(len(idx)):
          
            save_template = os.path.join(fakedir, filename)
            save_template = save_template.replace(".mp4", "")
            save_path = f"{save_template}_clip-{idx[j]:02d}.mp4"
            if os.path.exists(save_path):
                return True
    return False

def save_video_batch(samples, audios, filenames, fakedir, fps=12, loop=False, idx = [0]):
    # import ipdb; ipdb.set_trace()
    print("Video frames: ", samples[0][0].shape[2], "Audio frames: ", audios[0].shape[1], "FPS: ", fps)
    
    for i, filename in enumerate(filenames):
        for j in range(len(idx)):
            video = samples[j][0]
            audio_length = audios[j].shape[1]
            # b,c,t,h,w
            video = video.detach().cpu()

            video = torch.clamp(video.float(), -1., 1.)
            video = (video + 1.0) / 2.0
            video = (video * 255).to(torch.uint8).permute(1, 2, 3, 0)
            save_template = os.path.join(fakedir, filename)
            save_template = save_template.replace(".mp4", "")
            save_path = f"{save_template}_clip-{idx[j]:02d}.mp4"
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torchvision.io.write_video(save_path, video, audio_array = audios[j], audio_fps=16000,
                                        fps=fps, audio_codec='aac', video_codec='h264', options={'crf': '10'})
        
        # video = samples[i][0]
        # # b,c,t,h,w
        # video = video.detach().cpu()

        # video = torch.clamp(video.float(), -1., 1.)
        # video = (video + 1.0) / 2.0
        # video = (video * 255).to(torch.uint8).permute(1, 2, 3, 0)
        # save_template = os.path.join(fakedir, filename)
        # save_template = save_template.replace(".mp4", "")
        # save_path = f"{save_template}_clip-{idx[i]:02d}.mp4"
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # torchvision.io.write_video(save_path, video, fps=fps, video_codec='h264', options={'crf': '10'})

def save_video(samples, filename, fakedir, fps=12, loop=False, idx = 0):
    # prompt = prompt[0] if isinstance(prompt, list) else prompt

    video = samples[0][0]
    # b,c,t,h,w
    video = video.detach().cpu()

    video = torch.clamp(video.float(), -1., 1.)
    video = (video + 1.0) / 2.0
    video = (video * 255).to(torch.uint8).permute(1, 2, 3, 0)
    save_template = os.path.join(fakedir, filename)
    save_template = save_template.replace(".mp4", "")
    save_path = f"{save_template}_clip-{idx:02d}.mp4"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torchvision.io.write_video(save_path, video, fps=fps, video_codec='h264', options={'crf': '10'})



@torch.no_grad()
def run_inference(args, gpu_num=1, gpu_no=0):
    ## model config
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    data_config = config.pop("data", OmegaConf.create())
    
    ## set use_checkpoint as False as when using deepspeed, it encounters an error "deepspeed backend not set"
    model_config['params']['unet_config']['params']['use_checkpoint'] = False
    model = instantiate_from_config(model_config)
    model = model.to("cuda")
    model.perframe_ae = args.perframe_ae
    # assert os.path.exists(args.checkpoint), "Error: checkpoint Not Found!"
    model = load_model_checkpoint(model, args.checkpoint)
    model.eval()
    
    batch_size = data_config['params']['batch_size']
    # batch_size = 1
    
    ## run over data
    assert (args.height % 16 == 0) and (args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"
    # assert args.bs == 1, "Current implementation only support [batch size = 1]!"
    ## latent noise shape
    h, w = args.height // 8, args.width // 8
    
    
    channels = model.model.diffusion_model.out_channels
    n_frames = args.video_length
    print(f'Inference with {n_frames} frames')
    noise_shape = [batch_size*3, channels, n_frames, h, w]

    fakedir = os.path.join(args.exp_root, "samples")
    fakedir_separate = os.path.join(args.exp_root, "ASVA")
    os.makedirs(fakedir_separate, exist_ok=True)

    data_dir = data_config['params']['validation']['params']['data_dir']

    with open(data_config['params']['validation']['params']['filename_list']) as f:
        filename_list = [line.rstrip("\n") for line in f.readlines()]
    categories = set([filename.split("/")[0] for filename in filename_list])
    
    
    # 3. Generating
    # import ipdb; ipdb.set_trace()
    
    # for filename, category in tqdm(zip(filenames, categories), total=len(filenames)):
    start_idx = math.ceil(len(filename_list) / gpu_num) * gpu_no
    end_idx = min(math.ceil(len(filename_list) / gpu_num) * (gpu_no + 1), len(filename_list))

    steps = math.ceil((end_idx - start_idx) / batch_size) 
    for step in tqdm(range(steps),  desc='GPU %d; Sample %d - %d'%(gpu_no, start_idx, end_idx)):

        batch_filename= filename_list[start_idx:start_idx+batch_size]
        batch_filename_neg = []
        if args.swap_audio:
            for filename in batch_filename:
                category = filename.split("/")[0]
                negative_category = random.choice(list(categories - set([category])))
                negative_filename = random.choice([f for f in filename_list if f.startswith(negative_category)])
                batch_filename_neg.append(negative_filename)
        
        sub_filename_list, data_list, caption_list, audio_list, raw_audios, raw_videos, keyframes, frame_strides, keyframe_clips = \
            load_data_batch(data_dir, batch_filename, args.keyframe_gen_dir,  video_size=(args.height, args.width), video_frames=n_frames, video_fps=args.video_fps, fps_condition_type = model.fps_condition_type, swap_audio=args.swap_audio, batch_filename_neg=batch_filename_neg)
        videos = torch.cat(data_list, dim=0).to("cuda")
        audios = torch.cat(audio_list, dim=0).to("cuda")
        

        # if cuda out of memory, run sample by sample, batchsize=1
        n_samples_batch = videos.shape[0]
        batch_samples = []
        for i in range(n_samples_batch):
            noise_shape[0] = 1
            exists = check_exist(None, [raw_audios[i]], batch_filename, fakedir_separate, fps=args.video_fps, loop=False, idx=[i])
            if exists:
                print(f"Skip {batch_filename}, {i}")
                continue
            batch_samples_i = image_guided_synthesis(model, [caption_list[i]], videos[i].unsqueeze(0), audios[i].unsqueeze(0), keyframe_clips[i], \
                                        keyframes[i].unsqueeze(0), frame_strides[i].unsqueeze(0), \
                                        noise_shape, 1, args.ddim_steps, args.ddim_eta, args.inpainting_end_step, \
                                        args.unconditional_guidance_scale, args.cfg_img, args.cfg_audio, args.video_fps, args.text_input, args.multiple_cond_cfg, args.loop, args.interp, args.timestep_spacing, args.guidance_rescale)
            save_video_batch(batch_samples_i, [raw_audios[i]], batch_filename, fakedir_separate, fps=args.video_fps, loop=False, idx=[i])
        

        start_idx += batch_size

   