import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import MSELoss
import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from transformers import Wav2Vec2Config, Wav2Vec2Model

# from datasets import load_dataset
import datasets

from model import init_wav2vec, init_imagebind, ResNet, FCModel, CosSimilarityLoss, ScoreModel_mel_v2, ScoreModel_mel_KP, ScoreModel_mel_transformer
from audiodataset import AudioDataset
import matplotlib.pyplot as plt

from tqdm import tqdm
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import ModelCheckpoint


def main():

    # init data

    exp_name = 'keypoint_more_token'

    root = "/dockerx/share/DynamiCrafter/data/AVSync15/train"
    label = "/dockerx/share/DynamiCrafter/data/AVSync15/train_curves_npy"
    # vgg_sound = '/dockerx/local/data/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'
    # vgg_label = '/dockerx/local/data/VGGSound_audio_scores/label'
    train_dataset = AudioDataset(root_dir=root, label_dir=label, wav2vec_processor = None, split = "train")
    val_dataset = AudioDataset(root_dir=root, label_dir=label, wav2vec_processor = None, split = "test")
    # vgg_sound_trainset =AudioDataset(root_dir=vgg_sound, label_dir=vgg_label, wav2vec_processor = None, split = "vgg_sound")


    # Create a DataLoader to load the dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    # vgg_sound_train_dataloader = torch.utils.data.DataLoader(vgg_sound_trainset, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)


    
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6"
    
    # model = ScoreModel_mel_v2()
    # model = ScoreModel_mel_KP()
    model = ScoreModel_mel_transformer()

    tb_logger = pl_loggers.TensorBoardLogger(save_dir=f"logs/{exp_name}")
    ddp_strategy=DDPStrategy(find_unused_parameters=True)


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'checkpoints/{exp_name}',
        filename='model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min',
    )


    n_devices = 4
    # 0,1,2,3
    # 4,5,6,7
    trainer = L.Trainer(accelerator="gpu", devices=[0,1,2,3,4,5,6,7], max_epochs=500, 
                        strategy = ddp_strategy, 
                        precision=16, 
                        gradient_clip_val=0.5,
                        logger=tb_logger,
                        log_every_n_steps=100,
                        enable_progress_bar=True,
                        resume_from_checkpoint='/dockerx/share/wav2vec/logs/keypoint_w_plot/lightning_logs/version_50/checkpoints/epoch=199-step=2000.ckpt',)

    trainer.fit(model, train_dataloader, val_dataloader)
    # trainer.fit(model, vgg_sound_train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()