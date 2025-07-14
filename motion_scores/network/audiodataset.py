from typing import List, Union, Literal, Tuple
import itertools
import os
from torch.utils.data import Dataset
import torchaudio
import os
import numpy as np
import math

import torch
from torch.utils.data import Dataset
import torchaudio
import torchvision
from torchvision import transforms
from imagebind.data import waveform2melspec
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

def waveform_to_melspectrogram(
        waveform: Union[np.ndarray, torch.Tensor],
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
        waveform_clip, sample_rate, num_mel_bins, target_length, 
        # cut_and_pad=False
    )

    normalize = transforms.Normalize(mean=mean, std=std)
 
    audio_clip = normalize(waveform_melspec)
    
    
    return audio_clip  # (1, freq, time)



class AudioDataset(Dataset):
    def __init__(self, root_dir, label_dir, wav2vec_processor = None, transforms=None, split="train", format_ = "wav"):
        self.root_dir = root_dir
        self.label_dir = label_dir

        self.split = split
        self.format = format_
        train_split, test_split = self._get_file_paths(all = (split == 'all'))

        if split == "train":
            self.file_paths = train_split
        elif split == "test":
            self.file_paths = test_split
        elif split == "vgg_sound":
            self.file_paths = self._get_vgg_sound_file_paths()
            split = "train"
        else:
            self.file_paths = train_split



        self.transforms = transforms
        self.wav2vec_processor = wav2vec_processor
        self.clip_length = 2 * 16000  # 2 seconds at 16kHz
        # self.target_length = int(self.clip_length / 320)
        self.target_length = 48

    def _get_file_paths(self, all=False):
        file_paths = []
        label_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".{}".format(self.format)):
                    category = os.path.basename(root)
                    base_name = os.path.basename(file)

                    if self.label_dir is not None:
                        label_path = os.path.join(self.label_dir, category, base_name.split(".")[0]+".npy")

                    else:
                        label_path = None

                    file_paths.append((os.path.join(root, file), label_path))

                    
        # file_paths = sorted(file_paths)
        if not all:
            np.random.seed(42)  # Set the seed for reproducibility
            shuffled_paths = np.random.permutation(file_paths)

            train_split, test_split = shuffled_paths[:int(0.8 * len(shuffled_paths))], shuffled_paths[int(0.8 * len(shuffled_paths)):]
        else:
            train_split = file_paths
            test_split = file_paths
        return train_split, test_split
    
    def _get_vgg_sound_file_paths(self):


        file_paths = []
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".mp4"):
                    if os.path.exists(os.path.join(self.label_dir, file.split(".")[0]+".npy")):
                        file_paths.append((os.path.join(root, file), os.path.join(self.label_dir, file.split(".")[0]+".npy")))

        # file_paths = sorted(file_paths)
        
        # np.random.seed(42)  # Set the seed for reproducibility
        # shuffled_paths = np.random.permutation(file_paths)

        # train_split, test_split = file_paths[:int(0.8 * len(file_paths))], file_paths[int(0.8 * len(file_paths)):]
        train_split = file_paths
        return train_split
    def __len__(self):
        return len(self.file_paths)

    def getitem_no_lable(self, idx):

        file_path, label_path = self.file_paths[idx]

        if file_path.endswith(".mp4"):
            video, waveform, info = torchvision.io.read_video(file_path, pts_unit="sec") # [250, 720, 1280, 3])
            sample_rate = info["audio_fps"]
        else:
            waveform, sample_rate = torchaudio.load(file_path)
        
        # resample
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
        category = os.path.basename(os.path.dirname(file_path))
        base_name = os.path.basename(file_path)

        # Randomly select start time
        audio_original_length = waveform.shape[1]

        if self.split == "train":
            if audio_original_length > self.clip_length:
                available_length = audio_original_length - self.clip_length
                start_time = np.random.randint(0, audio_original_length - self.clip_length)

            elif audio_original_length == self.clip_length:
                start_time = 0
            else:
                print(f"audio_original_length: {audio_original_length}")
                rand_idx = np.random.randint(0,self.__len__())
                return self.__getitem__(rand_idx)

                waveform = torch.nn.functional.interpolate(target_length=self.clip_length)(waveform)
                start_time = 0
                audio_original_length = waveform.shape[1]
        else:
            start_time = 0

            if audio_original_length < self.clip_length:
                waveform = torchaudio.transforms.Interpolate(target_length=self.clip_length)(waveform)
                audio_original_length = waveform.shape[1]

        end_time = start_time + self.clip_length

        # Extract waveform and label within the selected time range
        waveform = waveform[:, start_time:end_time]

        if self.transforms:
            waveform = self.transforms(waveform)
        # FFT
        waveform_mel = waveform_to_melspectrogram(waveform)
        waveform_mel = waveform_mel[:, :, :198]

        # label = label
        fake_label = self.normalize_label(self.interpolate_label(waveform_mel.mean(dim=(0, 1))))

        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
      

        return waveform_mel, [0], [0], [0], fake_label, sample_rate, file_path, waveform

    def __getitem__(self, idx):

        file_path, label_path = self.file_paths[idx]

        if label_path is None:
            return self.getitem_no_lable(idx)

        if file_path.endswith(".mp4"):
            video, waveform, info = torchvision.io.read_video(file_path, pts_unit="sec") # [250, 720, 1280, 3])
            sample_rate = info["audio_fps"]
        else:
            waveform, sample_rate = torchaudio.load(file_path)
        
        # resample
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
        category = os.path.basename(os.path.dirname(file_path))
        base_name = os.path.basename(file_path)

        
        # if os.path.exists(os.path.join(self.label_dir, category, base_name.split(".")[0]+".npy")):
        #     label_path = os.path.join(self.label_dir, category, base_name.split(".")[0]+".npy")
        # else:
        #     label_path = os.path.join(self.label_dir, base_name.split(".")[0]+".npy")
        if label_path is None:
            label = None
        else:
            label = np.load(label_path)
        
        # Randomly select start time
        audio_original_length = waveform.shape[1]

        if self.split == "train":
            if audio_original_length > self.clip_length:
                available_length = audio_original_length - self.clip_length
                start_time = np.random.randint(0, audio_original_length - self.clip_length)

            elif audio_original_length == self.clip_length:
                start_time = 0
            else:
                print(f"audio_original_length: {audio_original_length}")
                rand_idx = np.random.randint(0,self.__len__())
                return self.__getitem__(rand_idx)

                waveform = torch.nn.functional.interpolate(target_length=self.clip_length)(waveform)
                start_time = 0
                audio_original_length = waveform.shape[1]
        else:
            start_time = 0

            if audio_original_length < self.clip_length:
                waveform = torchaudio.transforms.Interpolate(target_length=self.clip_length)(waveform)
                audio_original_length = waveform.shape[1]

        end_time = start_time + self.clip_length

        start_index = math.ceil(start_time * len(label) / audio_original_length)
        end_index = math.ceil(end_time * len(label) / audio_original_length)
        
        # Extract waveform and label within the selected time range
        waveform = waveform[:, start_time:end_time]

        label = label[start_index:end_index]
        
        # Interpolate label to target length
        # keypoint = None
        label = self.interpolate_label(label)
        # Normalize label

        label = self.normalize_label(label)

        keypoint = self.keypoint_detection(label)
        keypoint = np.abs(keypoint)

        # keypoint_smoothed = gaussian_filter1d(keypoint, 1.5)
        keypoint_smoothed = keypoint
        
        if self.transforms:
            waveform = self.transforms(waveform)
        # FFT
        waveform_mel = waveform_to_melspectrogram(waveform)
        waveform_mel = waveform_mel[:, :, :198]

        # label = label
        fake_label = self.normalize_label(self.interpolate_label(waveform_mel.mean(dim=(0, 1))))
        # fake_label = self.interpolate_label(waveform_mel.mean(dim=(0, 1)))

        # add augmentation
        
        # if self.split == "train":
        #     if np.random.rand() < 0.5:
                
        #         waveform = torch.flip(waveform, dims=[1])
        #         waveform_mel = torch.flip(waveform_mel, dims=[2])
        #         label = np.flip(label).copy()
        #         fake_label = np.flip(fake_label).copy()

        #     if np.random.rand() < 0.2:

        #         waveform_mel += torch.ones_like(waveform_mel) * np.random.normal(0, 0.1)
        #         waveform_mel *= np.random.uniform(0.9, 1.1)
        
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        if self.wav2vec_processor:
            waveform = waveform.squeeze(0)
            input_values = self.wav2vec_processor(waveform.cpu().numpy(), return_tensors="pt", sampling_rate = 16000).input_values.squeeze(0)
            
            return input_values, fake_label, keypoint, label,  sample_rate, file_path

        return waveform_mel, label, keypoint_smoothed, keypoint, fake_label, sample_rate, file_path, waveform
    
    def precision(self, logits, label, threshould=2):
        precision = self._precision(logits, label, threshould)
        return precision
        recall = self._recall(logits, label, threshould)
        if precision + recall == 0:
            return 0.5
        return 2 * precision * recall / (precision + recall)
    
    def _precision(self, logits, label, threshould=2):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().detach().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().detach().numpy()
        gt_keypoint = self.keypoint_detection(gaussian_filter1d(label, 1.5))
        pred_keypoint = self.keypoint_detection(gaussian_filter1d(logits, 1.5), prominence=0.0)
        
        gt_keypoint_loc = np.where(gt_keypoint == 1)[0]
        pred_keypoint_loc = np.where(pred_keypoint == 1)[0]
        if len(gt_keypoint_loc) == 0:
            # plot
            return 0.5
        if len(pred_keypoint_loc) == 0:
            return 0.0
            
        matched = 0
        dist_matrix = np.zeros((len(gt_keypoint_loc), len(pred_keypoint_loc)))
        match_matrix = np.zeros((len(gt_keypoint_loc), len(pred_keypoint_loc)))
        for i, j in itertools.product(range(len(gt_keypoint_loc)), range(len(pred_keypoint_loc))):
            dist_matrix[i, j] = np.abs(gt_keypoint_loc[i] - pred_keypoint_loc[j])

        for i in range(len(gt_keypoint_loc)):
            min_dist = np.min(dist_matrix[i])
            if min_dist <= threshould:
                match_matrix[i, np.argmin(dist_matrix[i])] = 1
        matched = np.sum(match_matrix)
        precision = matched / len(gt_keypoint_loc)

        return precision
    def _recall(self, logits, label, threshould=2):
        if isinstance(logits, torch.Tensor):
            logits = logits.cpu().detach().numpy()
        if isinstance(label, torch.Tensor):
            label = label.cpu().detach().numpy()
        gt_keypoint = self.keypoint_detection(gaussian_filter1d(label, 1.5))
        pred_keypoint = self.keypoint_detection(gaussian_filter1d(logits, 1.5), prominence=0.0)
        
        gt_keypoint_loc = np.where(gt_keypoint == 1)[0]
        pred_keypoint_loc = np.where(pred_keypoint == 1)[0]
        if len(gt_keypoint_loc) == 0:
            return 0.0
        if len(pred_keypoint_loc) == 0:
            return 0.5
            
        matched = 0
        dist_matrix = np.zeros((len(gt_keypoint_loc), len(pred_keypoint_loc)))
        match_matrix = np.zeros((len(gt_keypoint_loc), len(pred_keypoint_loc)))
        for i, j in itertools.product(range(len(gt_keypoint_loc)), range(len(pred_keypoint_loc))):
            dist_matrix[i, j] = np.abs(gt_keypoint_loc[i] - pred_keypoint_loc[j])

        for i in range(len(pred_keypoint_loc)):
            min_dist = np.min(dist_matrix[:, i])
            if min_dist <= threshould:
                match_matrix[np.argmin(dist_matrix[:, i]), i] = 1
        matched = np.sum(match_matrix)
        recall = matched / len(pred_keypoint_loc)

        return recall



    def normalize_label(self, label):
        # Normalize label here
        if np.max(label) == np.min(label):
            return label - np.min(label)
        else:
            normalized_label = (label - np.min(label)) / (np.max(label) - np.min(label))
            return normalized_label

    def interpolate_label(self, label, keypoint=None):
        # Interpolate label to target length here
        # print(f"interpolate {len(label)} to {self.target_length}")
        label = label[1:]
        interpolated_label = np.interp(np.linspace(0, 1, self.target_length), np.linspace(0, 1, len(label)), label)
        if keypoint is not None:
            keypoint = keypoint[1:]
            index_pos = np.where(keypoint == 1)[0]
            index_neg = np.where(keypoint == -1)[0]
            interpolated_keypoint = np.zeros(self.target_length)
            for i in index_pos:
                new_pos = math.ceil(i * self.target_length / len(label))
                interpolated_keypoint[new_pos] = 1
            for i in index_neg:
                new_pos = math.ceil(i * self.target_length / len(label))
                interpolated_keypoint[new_pos] = -1
            return interpolated_label.astype(np.float32), interpolated_keypoint
        else:
            return interpolated_label.astype(np.float32)
            
    
    def keypoint_detection(self, label, prominence=0.1):
        '''
        1. smooth the label
        2. Find the local maximum and minimum
        3. Denote the local maximmum to 1 and local minimum to -1, other to 0
        '''
        smoothed_label = label
        # smoothed_label = np.convolve(label, np.ones(5)/5, mode='same')
        peak = find_peaks(smoothed_label, distance=5, prominence=prominence)[0]
        valley = find_peaks(-smoothed_label, distance=5, prominence=prominence)[0]

        keypoint = np.zeros_like(label)
        keypoint[peak] = 1
        keypoint[valley] = -1
        # print(f"keypoint: {keypoint}")
        return keypoint
        
class AudioDataset_clip(AudioDataset):
    def __init__(self, root_dir, label_dir, wav2vec_processor = None, transforms=None, split="train"):
        super().__init__(root_dir, label_dir, wav2vec_processor, transforms, split)
    def __getitem__(self, idx):
        file_path, label_path = self.file_paths[idx]

        if file_path.endswith(".mp4"):
            video, waveform, info = torchvision.io.read_video(file_path, pts_unit="sec") # [250, 720, 1280, 3])
            sample_rate = info["audio_fps"]
        else:
            waveform, sample_rate = torchaudio.load(file_path)
        
        # resample
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        
        category = os.path.basename(os.path.dirname(file_path))
        base_name = os.path.basename(file_path)

        label = np.load(label_path)
        
        # Randomly select start time
        audio_original_length = waveform.shape[1]

        if self.split == "train":
            if audio_original_length > self.clip_length:
                available_length = audio_original_length - self.clip_length
                start_time = np.random.randint(0, audio_original_length - self.clip_length)

            elif audio_original_length == self.clip_length:
                start_time = 0
            else:
                waveform = torchaudio.transforms.Interpolate(target_length=self.clip_length)(waveform)
                start_time = 0
                audio_original_length = waveform.shape[1]
        else:
            start_time = 0

            if audio_original_length < self.clip_length:
                waveform = torchaudio.transforms.Interpolate(target_length=self.clip_length)(waveform)
                audio_original_length = waveform.shape[1]

        end_time = start_time + self.clip_length

        start_index = math.ceil(start_time * len(label) / audio_original_length)
        end_index = math.ceil(end_time * len(label) / audio_original_length)
        
        # Extract waveform and label within the selected time range
        waveform = waveform[:, start_time:end_time]

        label = label[start_index:end_index]
        # label = self.interpolate_label(label)
        label = self.normalize_label(label)

        keypoint = self.keypoint_detection(label)
        keypoint = np.abs(keypoint)
        # keypoint_smoothed = gaussian_filter1d(keypoint, 1.5)
        keypoint_smoothed = keypoint
        
        if self.transforms:
            waveform = self.transforms(waveform)
        # FFT
        waveform_mel = waveform_to_melspectrogram(waveform)
        waveform_mel = waveform_mel[:, :, :198]


        label = label
        # fake_label = self.normalize_label(self.interpolate_label(waveform_mel.mean(dim=(0, 1))))
        fake_label = self.interpolate_label(waveform_mel.mean(dim=(0, 1)))
        
        if waveform.ndim == 2:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if self.wav2vec_processor:
            waveform = waveform.squeeze(0)
            input_values = self.wav2vec_processor(waveform.cpu().numpy(), return_tensors="pt", sampling_rate = 16000).input_values.squeeze(0)
            
            return input_values, fake_label, keypoint, label,  sample_rate, file_path


        # rand_index = np.random.randint(0, len(keypoint))
        # start_point, end_point = self.get_start_end_point(waveform_mel, keypoint, rand_index)
        # waveform_mel = waveform_mel[:, :, start_point:end_point]
        # keypoint = keypoint[rand_index]

        # # resize to (1, 224 224) and normalize
        # waveform_mel = torch.nn.functional.interpolate(waveform_mel.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False).squeeze(0)

  
        return waveform_mel, label, keypoint_smoothed, keypoint, fake_label, sample_rate, file_path, waveform



    def get_start_end_point(self, waveform_mel, label, label_index):
        '''
        Get the start and end point for each label_index
        '''
        waveform_length = waveform_mel.shape[2]
        label_length = len(label)
        
        start_point = label_index * waveform_length // label_length
        end_point = (label_index + 1) * waveform_length // label_length
        return start_point, end_point

def plot_waveform(waveform, sample_rate, axis, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    axis.plot(time_axis, waveform.mean(axis=0), linewidth=1)
    if xlim:
        axis.set_xlim(xlim)
    if ylim:
        axis.set_ylim(ylim)
    
    # if num_channels == 1:
    #     axes = [axes]
    # else:
    #     # mean
    #     waveform = [waveform.mean(axis=0)]
    # for c in range(1):
    #     axes[c].plot(time_axis, waveform[c], linewidth=1)
    #     axes[c].grid(True)
    #     if num_channels > 1:
    #         axes[c].set_ylabel(f'Channel {c+1}')
    #     if xlim:
    #         axes[c].set_xlim(xlim)
    #     if ylim:
    #         axes[c].set_ylim(ylim)
    # figure.suptitle(title)
    # plt.title("")
    # plt.axis('off')

    # # Apply tight layout
    # plt.tight_layout(pad=0)  # Set pad=0 for tightest layout
    # return figure, axes


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Create an instance of the AudioDataset
    # root = '/dockerx/local/data/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video'
    # label = '/dockerx/local/data/VGGSound_audio_scores/label'
    # dataset = AudioDataset_clip(root_dir=root, label_dir=label, split="vgg_sound")
    root = '/dockerx/share/DynamiCrafter/data/AVSync15/train'
    label = '/dockerx/local/DynamiCrafter/data/AVSync15/train_curves_npy'
    dataset = AudioDataset_clip(root_dir=root, label_dir=label, split="test")

    # Create a DataLoader to load the dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    # Iterate over the dataloader to load the data in batches
    for batch in dataloader:
        waveform_mel, labels, keypoint_smoothed, keypoint, _ ,sample_rates, file_paths, waveform = batch

        # print(waveform_mel.shape)
        # figure, axes = plot_waveform(waveforms, sample_rates[0])
        # axes[1].plot(labels[0])

        # figure, axes = plt.subplots(3, 1)
        # axes[0].plot(waveforms_mel[0].mean(axis=(0,1)))
        # axes[1].plot(labels[0])
        # axes[1].scatter(np.where(keypoint[0] == 1), labels[0][np.where(keypoint[0] == 1)], color='r', marker='x')
        # axes[1].scatter(np.where(keypoint[0] == -1), labels[0][np.where(keypoint[0] == -1)], color='g', marker='x')
        # axes[1].plot(keypoint_smoothed[0], color='y')
        # plot_waveform(waveform[0], sample_rates[0], axes[2])
        
        # plt.savefig("label.png")    
        # print(file_paths[0])
        # import ipdb; ipdb.set_trace()

        # waveform_mel_norm = (waveforms_mel[0] - waveforms_mel[0].min()) / (waveforms_mel[0].max() - waveforms_mel[0].min())
        # waveform_mel_norm = waveform_mel_norm[0].numpy()
        
        # plt.imshow(waveform_mel_norm)
        # # plt.title(f"Label {int(keypoint[0])}")
        # plt.savefig("waveform_mel.png")
        # plt.close()
        # import ipdb; ipdb.set_trace()

        for i in range(len(keypoint)):

            label_vis = np.zeros_like(waveform_mel[i])

            for j in range(len(keypoint[i])):
                start_point, end_point = dataset.get_start_end_point(waveform_mel = waveform_mel[i], label = keypoint[i], label_index = j)
                # import ipdb; ipdb.set_trace()
                
                label_vis[:, :, start_point:end_point] = int(keypoint[i][j])
            waveform_mel = waveform_mel[i].numpy()
            # waveform_mel = (waveform_mel - np.min(waveform_mel) )/ ((np.max(waveform_mel) - np.min(waveform_mel)))
            waveform_mel = np.clip(waveform_mel, 0, 1)
            waveform_mel = np.concatenate([waveform_mel, label_vis], axis=1)
            plt.imshow(waveform_mel[0])
            plt.title(f"{file_paths[i]}")
            plt.savefig("waveform_mel.png")
            plt.close()
            import ipdb; ipdb.set_trace()
            