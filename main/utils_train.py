import os, re
from omegaconf import OmegaConf
import logging
mainlogger = logging.getLogger('mainlogger')

import torch
from collections import OrderedDict
from einops import rearrange, repeat

def init_workspace(name, logdir, model_config, lightning_config, rank=0):
    workdir = os.path.join(logdir, name)
    ckptdir = os.path.join(workdir, "checkpoints")
    cfgdir = os.path.join(workdir, "configs")
    loginfo = os.path.join(workdir, "loginfo")

    # Create logdirs and save configs (all ranks will do to avoid missing directory error if rank:0 is slower)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(loginfo, exist_ok=True)

    if rank == 0:
        if "callbacks" in lightning_config and 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
            os.makedirs(os.path.join(ckptdir, 'trainstep_checkpoints'), exist_ok=True)
        OmegaConf.save(model_config, os.path.join(cfgdir, "model.yaml"))
        OmegaConf.save(OmegaConf.create({"lightning": lightning_config}), os.path.join(cfgdir, "lightning.yaml"))
    return workdir, ckptdir, cfgdir, loginfo

def check_config_attribute(config, name):
    if name in config:
        value = getattr(config, name)
        return value
    else:
        return None

def get_trainer_callbacks(lightning_config, config, logdir, ckptdir, logger):
    default_callbacks_cfg = {
        "model_checkpoint": {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch}",
                "verbose": True,
                "save_last": False,
            }
        },
        "batch_logger": {
            "target": "callbacks.ImageLogger",
            "params": {
                "save_dir": logdir,
                "batch_frequency": 1000,
                "max_images": 4,
                "clamp": True,
            }
        },    
        "learning_rate_logger": {
            "target": "pytorch_lightning.callbacks.LearningRateMonitor",
            "params": {
                "logging_interval": "step",
                "log_momentum": False
            }
        },
        "cuda_callback": {
            "target": "callbacks.CUDACallback"
        },
    }

    ## optional setting for saving checkpoints
    monitor_metric = check_config_attribute(config.model.params, "monitor")
    if monitor_metric is not None:
        mainlogger.info(f"Monitoring {monitor_metric} as checkpoint metric.")
        default_callbacks_cfg["model_checkpoint"]["params"]["monitor"] = monitor_metric
        default_callbacks_cfg["model_checkpoint"]["params"]["save_top_k"] = 3
        default_callbacks_cfg["model_checkpoint"]["params"]["mode"] = "min"

    if 'metrics_over_trainsteps_checkpoint' in lightning_config.callbacks:
        mainlogger.info('Caution: Saving checkpoints every n train steps without deleting. This might require some free space.')
        default_metrics_over_trainsteps_ckpt_dict = {
            'metrics_over_trainsteps_checkpoint': {"target": 'pytorch_lightning.callbacks.ModelCheckpoint',
                                                   'params': {
                                                        "dirpath": os.path.join(ckptdir, 'trainstep_checkpoints'),
                                                        "filename": "{epoch}-{step}",
                                                        "verbose": True,
                                                        'save_top_k': -1,
                                                        'every_n_train_steps': 10000,
                                                        'save_weights_only': True
                                                    }
                                                }
        }
        default_callbacks_cfg.update(default_metrics_over_trainsteps_ckpt_dict)

    if "callbacks" in lightning_config:
        callbacks_cfg = lightning_config.callbacks
    else:
        callbacks_cfg = OmegaConf.create()
    callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)

    return callbacks_cfg

def get_trainer_logger(lightning_config, logdir, on_debug):
    default_logger_cfgs = {
        "tensorboard": {
            "target": "pytorch_lightning.loggers.TensorBoardLogger",
            "params": {
                "save_dir": logdir,
                "name": "tensorboard",
            }
        },
        "testtube": {
            "target": "pytorch_lightning.loggers.CSVLogger",
            "params": {
                    "name": "testtube",
                    "save_dir": logdir,
                }
            },
    }
    # os.makedirs('save', exist_ok=True)

    os.makedirs(os.path.join(logdir, "tensorboard"), exist_ok=True)
    default_logger_cfg = default_logger_cfgs["tensorboard"]
    if "logger" in lightning_config:
        logger_cfg = lightning_config.logger
    else:
        logger_cfg = OmegaConf.create()
    logger_cfg = OmegaConf.merge(default_logger_cfg, logger_cfg)
    return logger_cfg

def get_trainer_strategy(lightning_config):
    default_strategy_dict = {
        "target": "pytorch_lightning.strategies.DDPShardedStrategy"
    }
    if "strategy" in lightning_config:
        strategy_cfg = lightning_config.strategy
        return strategy_cfg
    else:
        strategy_cfg = OmegaConf.create()

    strategy_cfg = OmegaConf.merge(default_strategy_dict, strategy_cfg)
    return strategy_cfg

def interpolation_to_24(x):
    b, L, N = x.shape
    new_x = torch.zeros(b, 48*16, N)
    new_x_2 = torch.zeros(b, 24*16, N)
    #expand to 48 first
    for i in range(48):
        if i%3 == 0:
            new_x[:, i*16:i*16+16, :] = x[:, i//3*16:i//3*16+16, :]
        elif i<46:  
            alpha = 1- (i%3) / 3
            # print( i//3*16,i//3*16+16, i//3*16+32)
            new_x[:, i*16:i*16+16, :] = alpha * x[:, i//3*16:i//3*16+16, :] + (1-alpha) * x[:, i//3*16+16:i//3*16+32, :]
        else:
            new_x[:, i*16:i*16+16, :] = x[:, i//3*16:i//3*16+16, :]
    #expand to 24
    for i in range(24):
        new_x_2[:, i*16:i*16+16, :] = new_x[:, i*2*16:i*2*16+16, :]
    return new_x_2
    
    # three times
def load_checkpoints(model, model_cfg):
    if check_config_attribute(model_cfg, "pretrained_checkpoint"):
        pretrained_ckpt = model_cfg.pretrained_checkpoint
        resume_checkpoint = model_cfg.resume_checkpoint

        
        assert os.path.exists(pretrained_ckpt), "Error: Pre-trained checkpoint NOT found at:%s"%pretrained_ckpt
        mainlogger.info(">>> Load weights from pretrained checkpoint")

        pl_sd = torch.load(pretrained_ckpt, map_location="cpu")
        # 
        # import ipdb; ipdb.set_trace()
        
        if 'state_dict' in pl_sd.keys():
            # resume
            if resume_checkpoint:
                resume_pl_sd = torch.load(resume_checkpoint, map_location="cpu")
                model.load_state_dict(resume_pl_sd["state_dict"], strict=False)
                print(">>> Resume from checkpoint: %s"%resume_checkpoint)
            else:
                # # add: exampt the key from ImageBind
                
                img_proj_latent_weight = pl_sd['state_dict']['image_proj_model.latents']
                
                # print(img_proj_latent_weight.shape) # 1, 256, 1024
                reshape_target_length = model_cfg.params.image_proj_stage_config.params.video_length
                if img_proj_latent_weight.shape[1] == 16 * 16:
                    if reshape_target_length == 12:
                        img_proj_latent_weight = img_proj_latent_weight[:, :16*12, :]


                    elif reshape_target_length == 48:
                        img_proj_latent_weight = rearrange(img_proj_latent_weight, 'b (t l) d -> b d t l ', t = 16)
                        img_proj_latent_weight =  torch.nn.functional.interpolate(
                                                        img_proj_latent_weight, size=(reshape_target_length, 16), mode='bilinear', align_corners=False
                                                    )
                        img_proj_latent_weight = rearrange(img_proj_latent_weight, 'b d t l -> b (t l) d', t = reshape_target_length)
                        
                    print("Reshape image query shape", img_proj_latent_weight.shape) # 1, 256, 1024
                

                pl_sd_state_dict = {k: v for k, v in pl_sd['state_dict'].items() if not k.startswith('image_proj_model.latents')}
                pl_sd_state_dict['image_proj_model.latents'] = img_proj_latent_weight
                state_dict = model.state_dict()
                # for k, v in pl_sd_state_dict.items():
                #     # if k in state_dict.keys() and state_dict[k].shape == v.shape:
                #     if k in state_dict.keys():
                #         # if "2.transformer_blocks" in k:
                #         #     k = k.replace("2.transformer_blocks", "3.transformer_blocks")
                #         state_dict[k] = v
                model.load_state_dict(pl_sd_state_dict, strict=False)
                # model.load_state_dict(state_dict, strict=False)

            if model_cfg.finetune==1:
                freeze_layers = []
                not_freeze_layers = []
                for name, param in model.named_parameters():
                    if name in pl_sd["state_dict"].keys():  # Replace with the actual layer name
                        if name.startswith("image_proj_model") or '2.transformer_blocks' in name or 'temopral_conv' in name:
                            param.requires_grad = True
                            not_freeze_layers.append(name)
                        elif name.startswith("model.diffusion_model"):
                            if '2.transformer_blocks' in name or 'temopral_conv' in name:
                                param.requires_grad = True
                                not_freeze_layers.append(name)
                            for i in range(12):
                                if f'output_blocks.{i}.0' in name or f'output_blocks.{i}.1' in name:
                                    param.requires_grad = True
                                    not_freeze_layers.append(name)
                        
                        else:
                            freeze_layers.append(name)
                            param.requires_grad = False
                    else:
                        
                        if name.startswith("imagebind_model") and not name.startswith("imagebind_model.modality_preprocessors.audio"):
                            if 'modality_trunks.audio.blocks.0' in name or 'modality_trunks.audio.blocks.1' in name or  'modality_trunks.audio.blocks.2' in name:
                                param.requires_grad = True
                                not_freeze_layers.append(name)
                            else:
                                param.requires_grad = False
                        #         freeze_layers.append(name)
                        # if name.startswith("imagebind_model"):
                        #         param.requires_grad = False
                        #         freeze_layers.append(name)
                        else:
                            not_freeze_layers.append(name)
                            param.requires_grad = True
                    # print(f"Freeze {name}")
                log_text = "Freeze layers: %s\nNot freeze layers: %s"%(',\n'.join(freeze_layers), ',\n'.join(not_freeze_layers))
                with open("freeze_layers.txt", "w") as f:
                    f.write(log_text)
            elif model_cfg.finetune==2:
                freeze_layers = []
                not_freeze_layers = []
                for name, param in model.named_parameters():
                    if name in pl_sd["state_dict"].keys():  # Replace with the actual layer name
                        if name.startswith("image_proj_model") or '2.transformer_blocks' in name or '2.proj' in name or 'temopral_conv' in name:
                            param.requires_grad = True
                            not_freeze_layers.append(name)
                        elif 'fps_embedding' in  name:
                            param.requires_grad = True
                            not_freeze_layers.append(name)
                        else:
                            freeze_layers.append(name)
                            param.requires_grad = False
                    else:
                        
                        if name.startswith("imagebind_model") and not name.startswith("imagebind_model.modality_preprocessors.audio"):
                            param.requires_grad = False
                            freeze_layers.append(name)
                        #         freeze_layers.append(name)
                        # if name.startswith("imagebind_model"):
                        #         param.requires_grad = False
                        #         freeze_layers.append(name)
                        else:
                            not_freeze_layers.append(name)
                            param.requires_grad = True
                    # print(f"Freeze {name}")
                log_text = "Freeze layers: %s\nNot freeze layers: %s"%(',\n'.join(freeze_layers), ',\n'.join(not_freeze_layers))
                with open("freeze_layers.txt", "w") as f:
                    f.write(log_text)
            
            else:
                for name, param in model.named_parameters():
                    if name.startswith("imagebind_model"):
                        if 'modality_trunks.audio.blocks.0' in name or 'modality_trunks.audio.blocks.1' in name or  'modality_trunks.audio.blocks.2' in name or name.startswith("imagebind_model.modality_preprocessors.audio"):
                            param.requires_grad = True
                        else:
                            param.requires_grad = False
                freeze_layers = []
                not_freeze_layers = []
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        not_freeze_layers.append(name)
                    else:
                        freeze_layers.append(name)
                log_text = "Freeze layers: %s\nNot freeze layers: %s"%(',\n'.join(freeze_layers), ',\n'.join(not_freeze_layers))
                with open("freeze_layers.txt", "w") as f:
                    f.write(log_text)
                    
                    
            # for miss_key in missing_key:
            #     if not miss_key.startswith("imagebind"):
            #         raise KeyError("Kyes are missing")
            mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
        # try:
            
        #     if 'state_dict' in pl_sd.keys():
        #         # add: exampt the key from ImageBind
        #         # model.load_state_dict(resume_pl_sd["state_dict"], strict=False)
        #         img_proj_latent_weight = pl_sd['state_dict']['image_proj_model.latents']
                
        #         print(img_proj_latent_weight.shape) # 1, 256, 1024
        #         # img_proj_latent_weight = img_proj_latent_weight[:, :16*12, :]
        #         img_proj_latent_weight = interpolation_to_24(img_proj_latent_weight)
                
        #         print(img_proj_latent_weight.shape) # 1, 256, 1024

        #         state_dict = {k: v for k, v in pl_sd['state_dict'].items() if not k.startswith('image_proj_model.latents')}
        #         state_dict['image_proj_model.latents'] = img_proj_latent_weight
        #         model.load_state_dict(state_dict, strict=False)

        #         # for name, param in model.named_parameters():
        #         #     if name in pl_sd["state_dict"].keys():  # Replace with the actual layer name
        #         #         param.requires_grad = False
        #                 # print(f"Freeze {name}")
                    
        #         # for miss_key in missing_key:
        #         #     if not miss_key.startswith("imagebind"):
        #         #         raise KeyError("Kyes are missing")
        #         mainlogger.info(">>> Loaded weights from pretrained checkpoint: %s"%pretrained_ckpt)
        #     else:
        #         # deepspeed
        #         new_pl_sd = OrderedDict()
        #         for key in pl_sd['module'].keys():
        #             new_pl_sd[key[16:]]=pl_sd['module'][key]
        #         model.load_state_dict(new_pl_sd, strict=False)
        # except:
        #     model.load_state_dict(pl_sd)
    else:
        mainlogger.info(">>> Start training from scratch")

    return model

def set_logger(logfile, name='mainlogger'):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(logfile, mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s-%(levelname)s: %(message)s"))
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger