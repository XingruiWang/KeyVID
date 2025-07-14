import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torch.nn import MSELoss

from transformers import AutoTokenizer, AutoFeatureExtractor, AutoModelForCTC
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from transformers import Wav2Vec2Config, Wav2Vec2Model

# from datasets import load_dataset
import datasets

from model import init_wav2vec, init_imagebind, ResNet, FCModel, CosSimilarityLoss


from audiodataset import AudioDataset
import matplotlib.pyplot as plt

from tqdm import tqdm

def main():
    exp_name = "version-tmp"
    os.makedirs(f"vis-{exp_name}", exist_ok=True)
    os.makedirs(f"save-{exp_name}", exist_ok=True)
    # init model
    processor = None
    # processor, model = init_wav2vec()
    model = init_imagebind()
    # model = ResNet()
    # model = FCModel()
    
    
    # if torch.cuda.device_count() > 1:
    #     print("Using", torch.cuda.device_count(), "GPUs")
    #     model = nn.DataParallel(model)
    model = model.to("cuda")
    model.load_state_dict(torch.load("/dockerx/share/wav2vec/save/checkpoint-0-0.10344.pth"))
    
    # init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1.5e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    # init data
    root = "/dockerx/share/DynamiCrafter/data/AVSync15/train"
    label = "/dockerx/share/DynamiCrafter/data/AVSync15/train_curves_npy"
    vgg_sound = '/dockerx/local/data/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'
    vgg_label = '/dockerx/local/data/VGGSound_audio_scores/labels/label_9438'
    train_dataset = AudioDataset(root_dir=root, label_dir=label, wav2vec_processor = processor, split = "train")
    val_dataset = AudioDataset(root_dir=root, label_dir=label, wav2vec_processor = processor, split = "test")
    # vgg_sound_trainset =AudioDataset(root_dir=vgg_sound, label_dir=vgg_label, wav2vec_processor = processor, split = "vgg_sound")


    # Create a DataLoader to load the dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=8)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=8)
    # vgg_sound_train_dataloader = torch.utils.data.DataLoader(vgg_sound_trainset, batch_size=16, shuffle=True)
    n_epochs = 500

    mse_loss = MSELoss()
    cosine = CosSimilarityLoss()
    best_loss = 1000000
    best_AP = 0

    t = 0.2
    for epoch in range(n_epochs):
        # t will be used to balance the loss, start with 1, decrease to 0
        # t = max(0, 1 - epoch / 50)
        # train
        model.train()
        desc = f"Epoch {epoch}"
        tqdm_bar = tqdm(train_dataloader, desc=desc)

        n_count = 0
        AP = 0
        AP_3 = 0
        for i, (input_values, gt_score, keypoint_smoothed, keypoint, mel_score_seq, sample_rate, file_path, _) in enumerate(tqdm_bar):
            
            # forward sample through model to get greedily predicted transcription ids

            input_values = input_values.to("cuda")
            gt_score = gt_score.to("cuda")
            mel_score_seq = mel_score_seq.to("cuda") 
            logits = model(input_values, mel_score_seq)


            # sigmoid and then mse
            # loss = mse_loss(torch.sigmoid(logits), gt_score)
            # loss = mse_loss(logits, gt_score)
            zero = torch.tensor(0.0).to("cuda")            
            # loss = mse_loss(cosine(logits, gt_score), zero) * t + mse_loss(logits, gt_score) + mse_loss(cosine(logits, mel_score_seq), zero) * t
            loss = mse_loss(logits, gt_score) + mse_loss(logits, mel_score_seq) * 0.3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            with torch.no_grad():
                for n in range(logits.shape[0]):
                    AP += train_dataset.precision(logits[n], gt_score[n])
                    AP_3 += train_dataset.precision(logits[n], gt_score[n], threshould=3)
                    n_count += 1
            
            desc = f"Epoch {epoch} | LR {scheduler.get_last_lr()} | Iteration {i} | Loss {round(loss.item(), 5)} | AP: {round(AP / n_count, 4)} | AP@3: {round(AP_3 / n_count, 4)}"
            tqdm_bar.set_description(desc)


            # if i % 20 == 0:
            #     print(f"Epoch {epoch} | LR {scheduler.get_last_lr()} | Iteration {i} | Loss {loss.item()}")

        if epoch % 1 == 0:
            with torch.no_grad():
                plt.clf()
                # import ipdb; ipdb.set_trace()
                
                plt.plot(logits[0].cpu().numpy(), label='logits')
                plt.plot(gt_score[0].cpu().numpy(), label='label')
                plt.plot(mel_score_seq[0].cpu().numpy(), label='mel_score')

                plt.legend()

                plt.savefig(f"vis-{exp_name}/y_train.png")

                plt.clf()
                if len(input_values[0].shape) == 3:
                    plt.plot(input_values[0].cpu().numpy().mean(axis=(0,1)))
                else:
                    plt.plot(input_values[0].cpu().numpy())

                plt.savefig(f"vis-{exp_name}/x_train.png")
                plt.close()

        # evaluate
        model.eval()
        with torch.no_grad():
            mean_loss = 0
            count = 0
            n_count = 0
            AP = 0
            AP_3 = 0
            for i, (input_values, gt_score, keypoint_smoothed, keypoint, mel_score_seq, sample_rate, file_path, _) in enumerate(val_dataloader):
                input_values = input_values.to("cuda")
                gt_score = gt_score.to("cuda") 
                mel_score_seq = mel_score_seq.to("cuda")
                
                logits = model(input_values, mel_score_seq) 
                # logits = model(input_values).logits.squeeze(-1)
                # loss = mse_loss(torch.sigmoid(logits), gt_score)
                zero = torch.tensor(0.0).to("cuda")   
                loss = mse_loss(logits, gt_score) + mse_loss(cosine(logits, gt_score), zero)
                mean_loss += loss.item()
                count += 1

                for n in range(logits.shape[0]):
                    AP += train_dataset.precision(logits[n], gt_score[n])
                    AP_3 += train_dataset.precision(logits[n], gt_score[n], threshould=3)
                    n_count += 1

            mean_loss = mean_loss / count
            AP = AP / n_count
            AP_3 = AP_3 / n_count

            if mean_loss < best_loss:
                best_loss = mean_loss
                try:
                    torch.save(model.module.state_dict(), f"save-{exp_name}/best_model.pth")
                except:
                    torch.save(model.state_dict(), f"save-{exp_name}/best_model.pth")
            if AP > best_AP:
                best_AP = AP
                try:
                    torch.save(model.module.state_dict(), f"save-{exp_name}/best_model_AP.pth")
                except:
                    torch.save(model.state_dict(), f"save-{exp_name}/best_model_AP.pth")
                print("Save best model")
            
            if epoch % 50 == 0 and epoch != 0:
                try:
                    torch.save(model.module.state_dict(), f"save-{exp_name}/checkpoint-{epoch}-{round(mean_loss, 5)}.pth")
                except:
                    torch.save(model.state_dict(), f"save-{exp_name}/checkpoint-{epoch}-{round(mean_loss, 5)}.pth")
                print(f"Save model at epoch {epoch}")
            print(f"Epoch {epoch} Validation loss: {round(mean_loss, 5)} | best_AP:{round(best_AP, 4)} | AP: {round(AP, 4)} | AP@3: {round(AP_3, 4)}")

            # plot
            # plot logits[0] and gt_score[0]
            if epoch % 1 == 0:
                plt.clf()
                plt.plot(logits[0].cpu().numpy(), label='logits')
                plt.plot(gt_score[0].cpu().numpy(), label='label')
                plt.plot(mel_score_seq[0].cpu().numpy(), label='mel_score')


                plt.title(f"file_path: {file_path[0]}")
                
                plt.legend()
                
                
                plt.savefig(f"vis-{exp_name}/y_test.png")

                
                plt.clf()
                if len(input_values[0].shape) == 3:
                    plt.plot(input_values[0].cpu().numpy().mean(axis=(0,1)))
                else:
                    plt.plot(input_values[0].cpu().numpy())

                plt.savefig(f"vis-{exp_name}/x_test.png")
                plt.close()

                # import ipdb; ipdb.set_trace()
            
        scheduler.step()


def inference():
    processor = None
    # processor, model = init_wav2vec()
    model = init_imagebind()

    
    model = model.to("cuda")
    # model.load_state_dict(torch.load("save-2/best_model_norm.pth"))
    # model.load_state_dict(torch.load("save-version3/best_model_AP.pth"))


    model.load_state_dict(torch.load("save/best_model_version1_with_skip.pth"))



    model.eval()

    root = "/dockerx/local/data/AVSync15/train"
    label = "/dockerx/local/data/AVSync15/train_curves_npy"
    # root = "/dockerx/local/data/AVSync15/test"
    # label = "/dockerx/local/data/AVSync15/test_curves_npy"

    val_dataset = AudioDataset(root_dir=root, label_dir=label, wav2vec_processor = processor, split = "test")

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1)
    cosine = CosSimilarityLoss()
    mse_loss = MSELoss()

    with torch.no_grad():
        mean_loss = 0
        count = 0
        AP_3 = 0
        AP = 0
        n_count = 0
        for i, (input_values, gt_score, keypoint_smoothed, keypoint, mel_score_seq, sample_rate, file_path, _) in enumerate(tqdm(val_dataloader)):
            input_values = input_values.to("cuda")
            gt_score = gt_score.to("cuda") 
            mel_score_seq = mel_score_seq.to("cuda")
            
            logits = model(input_values, mel_score_seq)

            zero = torch.tensor(0.0).to("cuda")   
            loss = mse_loss(logits, gt_score) + mse_loss(cosine(logits, gt_score), zero) * 0.2
            mean_loss += loss.item()
            count += 1

            for n in range(logits.shape[0]):
                AP += val_dataset.precision(mel_score_seq[n], gt_score[n])
                AP_3 += val_dataset.precision(mel_score_seq[n], gt_score[n], threshould=3)
                n_count += 1

            

            # plt.clf()
            # plt.plot(logits[0].cpu().numpy(), label='Predict')
            # plt.plot(gt_score[0].cpu().numpy(), label='Ground Truth')
            # # plt.plot(mel_score_seq[0].cpu().numpy(), label='mel_score')


            # # plt.title(f"file_path: {file_path[0]}")
            
            # plt.legend()
            
            # name = file_path[0].split("/")[-1].split(".")[0]
            # cate = file_path[0].split("/")[-2]
            
            # os.makedirs(f"vis-last-new/{cate}", exist_ok=True)
            # plt.savefig(f"vis-last-new/{cate}/{name}.png")

            
            # plt.clf()
            # plt.close()

            # save npy
            # os.makedirs("prediction/motion", exist_ok=True)
            # os.makedirs("prediction/audio", exist_ok=True)


            # output_name = f'prediction/motion/{file_path[0].split("/")[-1].split(".")[0]}.npy'
            # np.save(output_name, logits[0].cpu().numpy())


            # output_name = f'prediction/audio/{file_path[0].split("/")[-1].split(".")[0]}.npy'
            # np.save(output_name, mel_score_seq[0].cpu().numpy())

            
        mean_loss = mean_loss / count
     
        print(f"Validation loss: {mean_loss}")
        print(f"AP: {AP / n_count}")
        print(f"AP@3: {AP_3 / n_count}")


def predict():
  
    model = init_imagebind()

    model = model.to("cuda")

    model.load_state_dict(torch.load("save/best_model_version1_with_skip.pth"))

    model.eval()

    root = "/dockerx/share/Dynamicrafter_audio/save/asva/asva_48/epoch=579-step=13920-inpainting_audio_7.5_img_2.0/ASVA"
    label = None

    val_dataset = AudioDataset(root_dir=root, label_dir=None, wav2vec_processor = None, split = "all", format_="mp4")

    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=1)
    cosine = CosSimilarityLoss()
    mse_loss = MSELoss()

    with torch.no_grad():
        mean_loss = 0
        count = 0
        AP_3 = 0
        AP = 0
        n_count = 0
        for i, (input_values, gt_score, keypoint_smoothed, keypoint, mel_score_seq, sample_rate, file_path, _) in enumerate(tqdm(val_dataloader)):
            



            name = file_path[0].split("/")[-1].split(".mp4")[0]
            cate = file_path[0].split("/")[-2]

            # save plot

            input_values = input_values.to("cuda")
            mel_score_seq = mel_score_seq.to("cuda")
            
            logits = model(input_values, mel_score_seq)

            plt.clf()
            # plt.plot(logits[0].cpu().numpy(), label='logits')
            # plt.plot(mel_score_seq[0].cpu().numpy(), label='mel_score')
            # plt.clf()
            plt.plot(logits[0].cpu().numpy(), label='Predict')
            plt.plot(gt_score[0].cpu().numpy(), label='GT')

        # plt.title(f"file_path: {file_path[0.split('/')[-1]}")
        
            plt.legend()
            

            plt.savefig(f"vis-last-new/{name}.png")

        
        
            plt.close()

            # # save npy
            # os.makedirs(f"prediction/motion/{cate}", exist_ok=True)
            # os.makedirs(f"prediction/audio/{cate}", exist_ok=True)


            # output_name = f'prediction/motion/{cate}/{name}.npy'
            # np.save(output_name, logits[0].cpu().numpy())


            # output_name = f'prediction/audio/{cate}/{name}.npy'
            # np.save(output_name, mel_score_seq[0].cpu().numpy())


if __name__ == "__main__":
    # predict()
    inference()
    # main()