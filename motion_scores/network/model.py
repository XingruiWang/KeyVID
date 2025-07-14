
import io
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torchaudio
import torch.nn as nn
import lightning as L
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC, Wav2Vec2Processor

from transformers import Wav2Vec2Config, Wav2Vec2Model
from einops import rearrange, repeat

from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformer import TransformerDecoderLayer, TransformerDecoder
from tqdm import tqdm

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class CosSimilarityLoss(nn.Module):
    def __init__(self):
        super(CosSimilarityLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, output, target):
        output_n = nn.functional.normalize(output, dim=1)
        target_n = nn.functional.normalize(target, dim=1)
        cos_loss = (1 - torch.cosine_similarity(output_n, target_n)).mean()

        zeros = torch.zeros_like(cos_loss)

        return self.mse(cos_loss, zeros) + self.mse(output, target)

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(512, 99)
    
    def forward(self, x):
        x = self.resnet(x)
        return x

class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc = nn.Linear(204, 99)
    
    def forward(self, x):
        x = torch.mean(x, dim=(1, 2))
        x = self.fc(x)
        return x

class ScoreModel(nn.Module):
    def __init__(self, wav2vec2CTC):

        super(ScoreModel, self).__init__()
        self.wav2vec2CTC = wav2vec2CTC
        self.linear = nn.Linear(16 * 99, 99)
    
    def forward(self, input_values):

        logits = self.wav2vec2CTC(input_values).logits.squeeze(-1)
        logits = rearrange(logits, 'b n c -> b (n c)')
        
        return self.linear(logits)

# class keypointAcuracy(torchmetrics.Metric):
#     def __init__(self, threshould = 0.5):
#         super(keypointAcuracy, self).__init__()
#         self.threshould = threshould
#         self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
#         self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

#     def update(self, pred, target):
#         pred = find_peaks(pred)


class ScoreModel_mel(nn.Module):
    def __init__(self, imagebindMoodel, model_set='1'):
        super(ScoreModel_mel, self).__init__()
        self.set = model_set
        self.imagebindMoodel = imagebindMoodel

        for param in self.imagebindMoodel.parameters():
            param.requires_grad = True

        # self.conv1 = nn.Conv1d(768, 32, kernel_size = 3)
        
        # For version 1 and 2

        if self.set == '1':
            self.linear1 = nn.Linear(1334, 48)
            self.linear2 = nn.Linear(768, 16)
            self.linear3 = nn.Linear(16*48, 48)
        elif self.set == '2':
            self.linear1 = nn.Linear(552, 46)
            self.linear2 = nn.Linear(768, 16)
            self.linear3 = nn.Linear(16*46, 46)
        elif self.set == '3':
            
            self.linear1 = nn.Linear(768, 16)
            self.linear2 = nn.Linear(16*12, 64)
            self.linear3 = nn.Linear(64, 1)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        # self.linear4 = nn.Linear(48, 48)

        self.ada_weight = nn.Parameter(torch.ones(1))
        self.ada_bias = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, audio, label = None):
        
        logits = self.imagebindMoodel.forward_audio(audio) # 229, 768,/ 4, 1334, 768

        if self.set in ['1', '2']:
            logits = rearrange(logits, 'b t c -> b c t') # 768, 229
            logits = self.linear1(logits) # 768 99 (downsample time axis)

            logits = rearrange(logits, 'b c t -> b t c') # 99 768

            out = self.linear2(self.relu1(logits)) # 99 16 downsample feature axis
            out = self.linear3(self.relu2(out.view(out.size(0), -1))) # 99 1

            # if label is not None:

                # return out.squeeze(-1) + logits.mean(dim=-1) + t * self.dropout(label) * self.ada_weight + self.ada_bias
        else:
            logits = rearrange(logits, 'b (h t) c -> b h t c', t = 46) # 4, 29, 16, 768
            logits = rearrange(logits, 'b h t c -> b t h c') # 4, 46, 16*29
            logits = self.linear1(logits)

            logits = rearrange(logits, 'b t h c -> b t (h c)') # 4, 46, 16*29
            out = self.linear2(self.relu1(logits))
            out = self.linear3(self.relu2(out))

        # return out.squeeze(-1)
        return out.squeeze(-1) + logits.mean(dim=-1)

        

class ScoreModel_mel_v2(L.LightningModule):
    def __init__(self):
        super(ScoreModel_mel_v2, self).__init__()
        self.imagebindMoodel = imagebind_model.imagebind_huge(pretrained=True)

        for param in self.imagebindMoodel.parameters():
            param.requires_grad = True

        # self.conv1 = nn.Conv1d(768, 32, kernel_size = 3)
        
        self.linear1 = nn.Linear(229, 99)
        self.linear2 = nn.Linear(768, 16)
        self.linear3 = nn.Linear(16*99, 99)
        self.linear4 = nn.Linear(99, 99)

        # self.dropout = nn.Dropout(0.2)
        self.ada_weight = nn.Parameter(torch.ones(1))
        self.ada_bias = nn.Parameter(torch.zeros(1))
        self.loss = CosSimilarityLoss()

        
    def forward(self, audio, label = None):
        
        logits = self.imagebindMoodel.forward_audio(audio) # 229, 768
        # logits = self.conv1(logits)
        logits = rearrange(logits, 'b t c -> b c t') # 768, 229
        logits = self.linear1(logits) # 768 99 (downsample time axis)
        logits = rearrange(logits, 'b c t -> b t c') # 99 768
        # drop out
        # logits = self.dropout(logits)

        out = self.linear2(logits) # 99 16 downsample feature axis
        out = self.linear3(out.view(out.size(0), -1)) # 99 1
        return out.squeeze(-1) + logits.mean(dim=-1)


        if mel_avg is not None:

            return out.squeeze(-1) + logits.mean(dim=-1) + mel_avg * self.ada_weight + self.ada_bias


        return out.squeeze(-1) + logits.mean(dim=-1)
    
    def training_step(self, batch, batch_idx):
        input_values, gt_score, label, sample_rate, file_path, _ = batch
        output = self(input_values, label)
        loss = self.loss(output, gt_score)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # sch = self.lr_schedulers()
        # if batch_idx % 1 == 0:
        #     print(sch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_values, gt_score, label, sample_rate, file_path, _ = batch
        output = self(input_values, label)
        loss = self.loss(output, gt_score)
        self.log('val_loss', loss, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3, weight_decay=0.1, betas=(0.9, 0.95))
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value



class ScoreModel_mel_KP(L.LightningModule):
    def __init__(self, num_decoder_layers=4, d_model=768, nhead=4, dim_feedforward=128, dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
        super(ScoreModel_mel_KP, self).__init__()
        # self.imagebindMoodel = imagebind_model.imagebind_huge(pretrained=True)
        self.imagebindMoodel = imagebind_model.ImageBindModel(
                audio_kernel_size=10,
                audio_stride=4,
                # out_embed_dim=768,
                vision_embed_dim=1280,
                vision_num_blocks=32,
                vision_num_heads=16,
                text_embed_dim=1024,
                text_num_blocks=24,
                text_num_heads=16,
                out_embed_dim=1024,
                audio_drop_path=0.1,
                imu_drop_path=0.7
            )
        for param in self.imagebindMoodel.parameters():
            param.requires_grad = True


        # self.conv1 = nn.Conv1d(768, 32, kernel_size = 3)
        
        self.linear1 = nn.Linear(229, 99)
        self.linear2 = nn.Linear(768, 16)
        self.linear3 = nn.Linear(16*99, 99)
        self.linear4 = nn.Linear(99, 99)

        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)


        # self.dropout = nn.Dropout(0.2)
        # self.ada_weight = nn.Parameter(torch.ones(1))
        # self.ada_bias = nn.Parameter(torch.zeros(1))
        # self.loss = nn.MSELoss()
        self.loss = nn.BCEWithLogitsLoss()


        
    def forward(self, audio, label = None):
        
        logits = self.imagebindMoodel.forward_audio(audio) # 229, 768
        # logits = self.conv1(logits)
        logits = rearrange(logits, 'b t c -> b c t') # 768, 229
        logits = self.linear1(logits) # 768 99 (downsample time axis)
        logits = rearrange(logits, 'b c t -> b t c') # 99 768
        # drop out
        # logits = self.dropout(logits)

        out = self.linear2(logits) # 99 16 downsample feature axis
        out = self.linear3(out.view(out.size(0), -1)) # 99 1

        # tgt = torch.zeros_like(query_embed)
        # out = self.decoder(logits, memory = logits)

        # return out.squeeze(-1) + self.dropout(logits.mean(dim=-1))
        return out.squeeze(-1) + logits.mean(dim=-1)
    
    def log_data(self, input_values, score,  gt_keypoint, gt_keypoint_smoothed, mel_score_seq, output_kpt, file_path, phase):
        print("log image data sigmoid ?? ", phase)
        score_np = score.cpu().numpy()
        gt_keypoint_np = gt_keypoint.cpu().numpy()
        gt_keypoint_smoothed_np = gt_keypoint_smoothed.cpu().numpy()
        output_kpt_np = torch.sigmoid(output_kpt).cpu().numpy()
        mel_score_seq_np = mel_score_seq.cpu().numpy()
        

        for i in tqdm(range(8)):

            # print(score_np.shape, gt_keypoint_np.shape, gt_keypoint_smoothed_np.shape, output_kpt_np.shape)
            plt.clf()
            plt.plot(score_np[i], label='motion_score')
            plt.scatter(np.where(gt_keypoint_np[i] == 1), score_np[i][np.where(gt_keypoint_np[i] == 1)], color='r', marker='x')
            plt.plot(output_kpt_np[i], label='pred_keypoint')
            plt.plot(gt_keypoint_smoothed_np[i], label='gt_keypoint_smoothed')
            # plt.plot(mel_score_seq_np[i], label='mel_score')
            plt.title(f"file: {file_path[i]}")
            plt.legend()
            plt.savefig(f"vis/y_{phase}_{i}.png")
        plt.close()


    def training_step(self, batch, batch_idx):
        input_values, score, keypoint_smoothed, keypoint, mel_score_seq, sample_rate, file_path, _ = batch
        output = self(input_values)

        # print(output.shape, keypoint.shape, output.dtype, keypoint.dtype)
        
        loss = self.loss(output, keypoint_smoothed.float())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=True, on_epoch=True, prog_bar=True, logger=True)

        # sch = self.lr_schedulers()
        if batch_idx == 0 and self.trainer.is_global_zero and self.current_epoch % 50 == 0:
            with torch.no_grad():
                self.log_data(input_values, score, keypoint, keypoint_smoothed, mel_score_seq, output, file_path, 'train')

        return loss
    
    def validation_step(self, batch, batch_idx):
        input_values, score, keypoint_smoothed, keypoint, mel_score_seq, sample_rate, file_path, _ = batch
        output = self(input_values)
        loss = self.loss(output, keypoint_smoothed.float())
        self.log('val_loss', loss, sync_dist=True)
        if batch_idx == 0 and self.trainer.is_global_zero and self.current_epoch % 50 == 0:
            self.log_data(input_values, score, keypoint, keypoint_smoothed, mel_score_seq, output, file_path, 'val')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.1, betas=(0.9, 0.95))
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=1e-6)
        return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]


    # def lr_scheduler_step(self, scheduler, metric):
    #     scheduler.step(epoch=self.current_epoch)  # timm's scheduler need the epoch value


class ScoreModel_mel_transformer(ScoreModel_mel_KP):
    def __init__(self, num_decoder_layers=4, d_model=768, nhead=4, dim_feedforward=128, dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
        super(ScoreModel_mel_transformer, self).__init__()
        # self.imagebindMoodel = imagebind_model.imagebind_huge(pretrained=True)

        # self.imagebindMoodel = imagebind_model.ImageBindModel(
        #         audio_kernel_size=10,
        #         audio_stride=4,
        #         audio_target_len=198,
        #         num_cls_tokens=0,
        #         # out_embed_dim=768,
        #         vision_embed_dim=1280,
        #         vision_num_blocks=32,
        #         vision_num_heads=16,
        #         text_embed_dim=1024,
        #         text_num_blocks=24,
        #         text_num_heads=16,
        #         out_embed_dim=1024,
        #         audio_drop_path=0.1,
        #         imu_drop_path=0.7
        #     )
        self.imagebindMoodel = imagebind_model.ImageBindModel(
                audio_kernel_size=10,
                audio_stride=4,
                audio_target_len=198,
                num_cls_tokens=0,
                # out_embed_dim=768,
                vision_embed_dim=1280,
                vision_num_blocks=32,
                vision_num_heads=16,
                text_embed_dim=1024,
                text_num_blocks=24,
                text_num_heads=16,
                out_embed_dim=1024,
                audio_drop_path=0.1,
                imu_drop_path=0.7
            )
        # model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))
        self.unstrick_load_ckpt(self.imagebindMoodel, ".checkpoints/imagebind_huge.pth")
        # for param in self.imagebindMoodel.parameters():
        #     param.requires_grad = True

        self.linear1 = nn.Linear(768, 64)

        self.out = MLP(64*30, 64, 1, 3)
        
        
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.backbone = backbone
        # self.aux_loss = aux_loss
    
    def unstrick_load_ckpt(self, module, ckpt):
        ckpt = torch.load(ckpt)
        state_dict = module.state_dict()
        unused = []
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
            else:
                unused.append(k)
        print(f"Unused keys: {unused}")
        module.load_state_dict(state_dict)

    def forward(self, audio, label = None):
        
        logits = self.imagebindMoodel.forward_audio(audio) # 229, 768
        logits = rearrange(logits, 'b (h w) c -> b h w c', w = 48) # 4, 30, 48, 768
        logits = logits.permute((0, 2, 1, 3)) # [4, 48, 30, 768])

        logits = self.linear1(logits) # 768 -> 64
        logits = rearrange(logits, 'b h w c -> b h (w c)') 
        
        out = self.out(logits) # 64 * 30 -> 1
        return out.squeeze(-1) # (48,)

class ScoreModel_mel_transformer2(ScoreModel_mel_v2):
    def __init__(self, num_decoder_layers=4, d_model=768, nhead=4, dim_feedforward=128, dropout=0.1, activation="relu", normalize_before=False, return_intermediate_dec=False):
        super(ScoreModel_mel_transformer, self).__init__()
        # self.imagebindMoodel = imagebind_model.imagebind_huge(pretrained=True)

        # self.imagebindMoodel = imagebind_model.ImageBindModel(
        #         audio_kernel_size=10,
        #         audio_stride=4,
        #         audio_target_len=198,
        #         num_cls_tokens=0,
        #         # out_embed_dim=768,
        #         vision_embed_dim=1280,
        #         vision_num_blocks=32,
        #         vision_num_heads=16,
        #         text_embed_dim=1024,
        #         text_num_blocks=24,
        #         text_num_heads=16,
        #         out_embed_dim=1024,
        #         audio_drop_path=0.1,
        #         imu_drop_path=0.7
        #     )
        self.imagebindMoodel = imagebind_model.ImageBindModel(
                audio_kernel_size=16,
                audio_stride=4,
                audio_target_len=198,
                num_cls_tokens=0,
                # out_embed_dim=768,
                vision_embed_dim=1280,
                vision_num_blocks=32,
                vision_num_heads=16,
                text_embed_dim=1024,
                text_num_blocks=24,
                text_num_heads=16,
                out_embed_dim=1024,
                audio_drop_path=0.1,
                imu_drop_path=0.7
            )
        # model.load_state_dict(torch.load(".checkpoints/imagebind_huge.pth"))
        self.unstrick_load_ckpt(self.imagebindMoodel, ".checkpoints/imagebind_huge.pth")
        # for param in self.imagebindMoodel.parameters():
        #     param.requires_grad = True

        self.linear1 = nn.Linear(229, 99)
        self.linear2 = nn.Linear(768, 16)
        self.linear3 = nn.Linear(16*99, 99)
        self.linear4 = nn.Linear(99, 99)

        # self.dropout = nn.Dropout(0.2)
        self.ada_weight = nn.Parameter(torch.ones(1))
        self.ada_bias = nn.Parameter(torch.zeros(1))
        self.loss = CosSimilarityLoss()
        # self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        # self.backbone = backbone
        # self.aux_loss = aux_loss
    
    def unstrick_load_ckpt(self, module, ckpt):
        ckpt = torch.load(ckpt)
        state_dict = module.state_dict()
        unused = []
        for k, v in ckpt.items():
            if k in state_dict and state_dict[k].shape == v.shape:
                state_dict[k] = v
            else:
                unused.append(k)
        print(f"Unused keys: {unused}")
        module.load_state_dict(state_dict)

    def forward(self, audio, label = None):
        
        logits = self.imagebindMoodel.forward_audio(audio) # 229, 768
        # logits = self.conv1(logits)
        logits = rearrange(logits, 'b t c -> b c t') # 768, 229
        logits = self.linear1(logits) # 768 99 (downsample time axis)
        logits = rearrange(logits, 'b c t -> b t c') # 99 768
        # drop out
        # logits = self.dropout(logits)

        out = self.linear2(logits) # 99 16 downsample feature axis
        out = self.linear3(out.view(out.size(0), -1)) # 99 1
        return out.squeeze(-1) + logits.mean(dim=-1)


        if mel_avg is not None:

            return out.squeeze(-1) + logits.mean(dim=-1) + mel_avg * self.ada_weight + self.ada_bias


        return out.squeeze(-1) + logits.mean(dim=-1)

def init_imagebind():
    imagebindMoodel = imagebind_model.ImageBindModel(
                audio_kernel_size=16,
                audio_stride=4,
                audio_target_len=198,
                num_cls_tokens=0,
                # out_embed_dim=768,
                vision_embed_dim=1280,
                vision_num_blocks=32,
                vision_num_heads=16,
                text_embed_dim=1024,
                text_num_blocks=24,
                text_num_heads=16,
                out_embed_dim=1024,
                audio_drop_path=0.1,
                imu_drop_path=0.7
            )

    ckpt = torch.load(".checkpoints/imagebind_huge.pth")
    state_dict = imagebindMoodel.state_dict()
    unused = []
    for k, v in ckpt.items():
        if k in state_dict and state_dict[k].shape == v.shape:
            state_dict[k] = v
        else:
            unused.append(k)
    print(f"Unused keys: {unused}")
    imagebindMoodel.load_state_dict(state_dict)

    # imagebindMoodel.load_state_dict(torch.load())
    # imagebindMoodel = imagebind_model.imagebind_huge(pretrained=True)
    
    model = ScoreModel_mel(imagebindMoodel)
    return model
    


def init_wav2vec():

    # def freeze_feature_encoder(self):
    #     """
    #     Calling this function will disable the gradient computation for the feature encoder so that its parameter will
    #     not be updated during training.
    #     """
    #     self.wav2vec2.feature_extractor._freeze_parameters()

    # def freeze_base_model(self):
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-base-960h")
    config.vocab_size = 16
    
    wav2vec2CTC = Wav2Vec2ForCTC(config)
    
    wav2vec2CTC.wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    model = ScoreModel(wav2vec2CTC)


    # model.freeze_feature_extractor()
    # for param in model.wav2vec2.parameters():
    #     param.requires_grad = False
    return processor, model