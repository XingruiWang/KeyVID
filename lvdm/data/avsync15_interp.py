import os
import random
from tqdm import tqdm
import pandas as pd
from decord import VideoReader, cpu
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import torch.nn.functional as F

import torchaudio
import ffmpeg
from pytorchvideo.data.clip_sampling import ConstantClipsPerVideoSampler
from fractions import Fraction 
from imagebind.data import waveform2melspec

import logging
import math
import random
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import random
logging.getLogger("requests").setLevel(logging.ERROR)

def transform_audio_data(
    waveform, sr,
    start, end, 
    # device,
    num_mel_bins=128,
    target_length=204,
    sample_rate=16000,
    clip_duration=2,
    clips_per_video=3,
    mean=-4.268,
    std=9.138,
):
    
    clip_sampler = ConstantClipsPerVideoSampler(
        clip_duration=clip_duration, clips_per_video=clips_per_video
    )

    if sample_rate != sr:
        waveform = torchaudio.functional.resample(
            waveform, orig_freq=sr, new_freq=sample_rate
        )

    # all_clips_timepoints = get_clip_timepoints(
    #     clip_sampler, waveform.size(1) / sample_rate
    # )
    # all_clips = []
    # if end - start != clip_duration:
    #     raise ValueError(f"{end} - {start} != {clip_duration}")

    clip_timepoints = (start, end)
    # for clip_timepoints in all_clips_timepoints:
    waveform_clip = waveform[
        :,
        int(clip_timepoints[0] * sample_rate) : int(
            clip_timepoints[1] * sample_rate
        ),
    ]
    # import ipdb; ipdb.set_trace()

    waveform_melspec = waveform2melspec(
        waveform_clip, sample_rate, num_mel_bins, target_length
    )
    # all_clips.append(waveform_melspec)
    normalize = transforms.Normalize(mean=mean, std=std)
    waveform_melspec = normalize(waveform_melspec)
    # audio_clip = normalize(waveform_melspec).to(device)
    # all_clips = [normalize(ac).to(device) for ac in all_clips]

    # all_clips = torch.stack(all_clips, dim=0)
    # audio_outputs.append(all_clips)
    # return torch.stack(audio_outputs, dim=0)
    return waveform_melspec, waveform_clip


class AVSync15(Dataset):
    """
    AVSync15 Dataset.
    Assumes AVSync15 data is structured as follows.
    AVSync15/
        train/
            label/      ($page_dir)
                aaa.mp4           (videoid.mp4)
                ...
                bb.mp4
            ...
    """
    def __init__(self,
                 data_dir,
                 keyframe_dir,
                 caption_dir=None,
                 subsample=None,
                 video_length=16,
                 resolution=[256, 512],
                 frame_stride=1,
                 frame_stride_min=1,
                 spatial_transform=None,
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=False,
                 fixed_fps=None,
                 random_fs=False,
                 flip = True,
                 keyframe_ratio = 0.5
                 ):
        # self.meta_path = meta_path
        self.data_dir = data_dir # AVSync15/train
        self.keyframe_dir = keyframe_dir
        self.caption_dir = caption_dir
        self.subsample = subsample
        self.video_length = video_length
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.fps_max = fps_max
        self.frame_stride = frame_stride
        self.frame_stride_min = frame_stride_min
        self.fixed_fps = fixed_fps
        self.load_raw_resolution = load_raw_resolution
        self.random_fs = random_fs
        self.flip = flip
        self.keyframe_ratio = keyframe_ratio

        if "VGGSound_final" in data_dir:
            # read csv

            self.caption = pd.read_csv(caption_dir)
            self._load_metadata_vgg(self.caption)
        else:
            self._load_metadata()
            self.caption = None


        if spatial_transform is not None:
            if spatial_transform == "random_crop":
                self.spatial_transform = transforms.RandomCrop(crop_resolution)
            elif spatial_transform == "center_crop":
                self.spatial_transform = transforms.Compose([
                    transforms.CenterCrop(resolution),
                    ])            
            elif spatial_transform == "resize_center_crop":
                # assert(self.resolution[0] == self.resolution[1])
                self.spatial_transform = transforms.Compose([
                    transforms.Resize(min(self.resolution)),
                    transforms.CenterCrop(self.resolution),
                    ])
            elif spatial_transform == "resize":
                self.spatial_transform = transforms.Resize(self.resolution)
            else:
                raise NotImplementedError
        else:
            self.spatial_transform = None
    
    def _load_metadata_vgg(self, caption):
        self.all_video_paths = []
        for index, row in caption.iterrows():
            name = row.iloc[0]
            idx = row.iloc[1]
            cap = row.iloc[2]
            video_name = f"{name}_{idx:06d}.mp4"
            if os.path.exists(os.path.join(self.data_dir,video_name)) and os.path.exists(os.path.join(self.keyframe_dir, video_name.split(".")[0] + ".npy")):
                self.all_video_paths.append((cap, video_name))
            # Perform your desired operation on the first column value
        print(f"Total {len(self.all_video_paths)} videos loaded.")

        # for i, p in enumerate(os.listdir(self.data_dir)):
        #     if p.endswith(".mp4"):
        #         idx = p.split(".")[0]
        #         filtered_row = df[df[0] == first_column_value]
        #         cap = caption[caption[0] == idx].iloc[0,1]
        #         self.all_video_paths.append(cap, p)

            
    
    def _load_metadata(self):
        self.all_video_paths = []
        cates = sorted(os.listdir(self.data_dir))
        self.all_cates = cates
        for i, cate in enumerate(cates):
            sub_path = os.path.join(self.data_dir, cate)
            video_path = sorted([p for p in os.listdir(sub_path) if p.endswith(".mp4")])
            video_path = [f"{cate}/{v}" for v in video_path]
            self.all_video_paths += video_path

    #     metadata = pd.read_csv(self.meta_path, dtype=str)
    #     print(f'>>> {len(metadata)} data samples loaded.')
    #     if self.subsample is not None:
    #         metadata = metadata.sample(self.subsample, random_state=0)
   
    #     metadata['caption'] = metadata['name']
    #     del metadata['name']
    #     self.metadata = metadata
    #     self.metadata.dropna(inplace=True)

    def _get_video_path(self, sample):
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp
    
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
        # smoothed_label = label
        smoothed_label = np.convolve(label, np.ones(5)/5, mode='same')
        peak = find_peaks(smoothed_label, distance=5, prominence=prominence)[0]
        valley = find_peaks(-smoothed_label, distance=5, prominence=prominence)[0]

        keypoint = np.zeros_like(label)
        keypoint[peak] = 1
        keypoint[valley] = -1
        # print(f"keypoint: {keypoint}")
        return keypoint

    def select_keyframe(self, keyframe, start_idx, target_length, target_fps, fps_ori):
        # print(keyframe, start_idx, target_length, target_fps, fps_ori)

        # 1. Random select 6 keyframes where the keypoint is 1
        # 2. Random select 1 keyframes where keypoint is -1 between each two keyframes
        # 3. Random select until the target length is reached for keypoint 0

        end_idx = start_idx + round(2 * fps_ori)
        keyframe = keyframe[start_idx:end_idx]

        pos_1 = np.where(keyframe == 1)[0]
        pos_neg_1 = np.where(keyframe == -1)[0]
        # print("Pos:", pos_1)
        # print("Neg:", pos_neg_1)

        if len(pos_1) == 0 or (len(pos_1) == 1 and pos_1[0] == 0):
            random_keypoint = [0, end_idx - start_idx - round(fps_ori / target_fps)]
        else:
            select_num = min(6, len(pos_1))
            random_peak = np.random.choice(pos_1, select_num, replace=False)
            random_valley = np.array([])
            if 0 not in random_peak:
                random_peak = np.insert(random_peak, 0, 0)
            else:
                select_num -= 1
            for i in range(0, select_num):

                peak_1, peak_2 = random_peak[i], random_peak[i+1]

                all_valley = pos_neg_1[(pos_neg_1 > peak_1) & (pos_neg_1 < peak_2)]
                if len(all_valley) > 0:
                    random_v = np.random.choice(all_valley, 1)
                    random_valley = np.concatenate([random_valley, random_v])
                else:
                    random_v = [(peak_1 + peak_2) // 2]
                    random_valley = np.concatenate([random_valley, random_v])
            if len(pos_neg_1) > 0 and pos_neg_1[-1] not in random_valley:
            # if pos_neg_1[-1] not in random_valley:
                random_valley = np.concatenate([random_valley, [pos_neg_1[-1]]])

            random_keypoint = np.sort(np.unique(np.concatenate([random_peak, random_valley])))
        
        # print(f"random_keypoint: {random_keypoint}")
        if len(random_keypoint) < target_length:
            # print("uniform sampling")
            idx_to_add = np.array([])
            num_padding = target_length - len(random_keypoint)

            start, intervel_len = random_keypoint[:-1], [random_keypoint[i+1] - random_keypoint[i] for i in range(len(random_keypoint)-1)]
            # print(f"start: {start}, intervel_len: {intervel_len}")
            # num_padding_interval = intervel_len * num_padding / sum(intervel_len)
            num_padding_interval = [num_padding * l / sum(intervel_len) for l in intervel_len]
            allocated = [int(w_share // 1) for w_share in num_padding_interval]
            sum_allocated = sum(allocated)
            remaining = num_padding - sum_allocated
            if remaining > 0:
                try:
                    fractions = [(num_padding_interval[i] - allocated[i], i) for i in range(len(allocated))]
                    fractions.sort(key=lambda x: x[0], reverse=True)
                    for i in range(remaining):
                        _, idx = fractions[i]
                        allocated[idx] += 1
                except:
                    print("===========================")
                    print(random_keypoint, num_padding_interval, pos_1,pos_neg_1, remaining)
                    assert False
            
            for i in range(len(intervel_len)):
                num_padding = allocated[i]
                idx_to_add = np.concatenate([idx_to_add, np.linspace(random_keypoint[i], random_keypoint[i+1], num_padding+1, endpoint=False).astype(int)[1:]])
            random_keypoint = np.sort(np.concatenate([random_keypoint, idx_to_add]))
        # print(f"after interpolation random_keypoint : {random_keypoint}")
        # assert len(random_keypoint) == target_length, f"random_keypoint={len(random_keypoint)}, target_length={target_length}"

        random_keypoint_condition = random_keypoint * 24 / fps_ori 
        random_keypoint_condition = np.round(random_keypoint_condition)
        # random_keypoint_condition = np.unique(np.round(random_keypoint_condition))
        # import ipdb; ipdb.set_trace()
        
        
        # Exnsure the length of keypoint is equal to target_length
        '''
        if len(random_keypoint_condition) > target_length:
            print("Random keypoint length is greater than target length")
            print("Random keypoint:", random_keypoint)
            random_keypoint = random_keypoint[:target_length]
        elif len(random_keypoint_condition) < target_length:
            print("Random keypoint length is less than target length")
            print("Random keypoint:", random_keypoint)
            unselected = set(range(end_idx - start_idx)) - set(random_keypoint)
            extra = np.random.choice(list(unselected), target_length - len(random_keypoint), replace=False)
            random_keypoint = np.sort(np.concatenate([random_keypoint, extra]))
        print(f"random_keypoint: {random_keypoint}")
        '''
        # if len(random_keypoint_condition) < target_length:
            # unselected = set(range(48)) - set(random_keypoint_condition)
            # extra = np.random.choice(list(unselected), target_length - len(random_keypoint_condition), replace=False)
            # random_keypoint_condition = np.sort(np.concatenate([random_keypoint_condition, extra]))
        if len(random_keypoint_condition) > target_length:

            random_keypoint = random_keypoint[:target_length]
            random_keypoint_condition = random_keypoint_condition[:target_length]
        
        frame_stride = random_keypoint_condition[1:] - random_keypoint_condition[:-1]
        frame_stride = np.insert(frame_stride, 0, frame_stride[0])
        
        
        return random_keypoint.astype(int)+start_idx, frame_stride.astype(float), random_keypoint_condition.astype(int)
    
        # return random_keypoint.astype(int), frame_stride.astype(float)
    

    def select_frame(self, start_idx, target_length, target_fps=24, fps_ori=30, index=0):
        # Random select 6 keyframes where the keypoint is 1
        end_idx = start_idx + round(2 * fps_ori)

        random_int = np.random.choice([0,1])
        if random_int == 0:
            random_keypoint = np.random.choice(range(1, end_idx - start_idx), target_length-1, replace=False)
            random_keypoint = np.sort(random_keypoint)
            random_keypoint = np.insert(random_keypoint, 0, 0)

        elif random_int == 1:
            # uniform select target_length keyframes between start_idx and end_idx
            random_fs = random.randint(1, int(2*fps_ori/target_length))

            if random_fs * target_length >= end_idx - start_idx:
                random_fs = (end_idx - start_idx) // target_length
            random_keypoint = np.array([t * random_fs for t in range(target_length)]).astype(float) 
        else:
            raise NotImplementedError
            
        random_keypoint_condition = random_keypoint * 24 / fps_ori 
        random_keypoint_condition = random_keypoint_condition.astype(int)

        
        frame_stride = random_keypoint_condition[1:] - random_keypoint_condition[:-1]
        frame_stride = np.insert(frame_stride, -1, frame_stride[-1])
        return random_keypoint.astype(int)+start_idx, frame_stride.astype(float), random_keypoint_condition.astype(int)
    
    def select_uniform_frame(self, start_idx, target_length, target_fps=30, fps_ori=30, index=0):
        # Random select 6 keyframes where the keypoint is 1
        end_idx = start_idx + round(2 * fps_ori)
        # random_fs = random.randint(1, int(2*fps_ori/target_length))
        random_fs = random.randint(1, min(2, int(2*fps_ori/target_length)))
        # random_fs = random.choice([1, int(2*fps_ori/target_length)])

        if random_fs * target_length >= end_idx - start_idx:
            random_fs = (end_idx - start_idx) / target_length
        random_keypoint = np.array([int(t * random_fs) for t in range(target_length)]).astype(float) 

        # end_padding = end_idx - start_idx - random_keypoint[-1]
        # if end_padding > 0:
        #     random_padding = np.random.choice(range(1, int(end_padding)), 1)
        #     random_keypoint += random_padding
        #     random_keypoint[0] = 0
        
        random_keypoint_condition = random_keypoint * 24 / fps_ori 
        random_keypoint_condition = random_keypoint_condition.astype(int)
        
        frame_stride = random_keypoint_condition[1:] - random_keypoint_condition[:-1]
        frame_stride = np.insert(frame_stride, -1, frame_stride[-1])
        
        return random_keypoint.astype(int)+start_idx, frame_stride.astype(float), random_keypoint_condition.astype(int)

    def __getitem__(self, index):
        if self.random_fs:
            frame_stride = random.randint(self.frame_stride_min, self.frame_stride)
        else:
            frame_stride = self.frame_stride

        sample = self.all_video_paths[index]
        cate, video_path = sample.split("/")
        

        video_path = os.path.join(self.data_dir, sample)
        keyframe_path = os.path.join(self.keyframe_dir, sample.split(".")[0] + ".npy")
        caption = " ".join(cate.split("_"))

        # ================== 1. Load videos ===================
        video, waveform, info = torchvision.io.read_video(video_path, pts_unit="sec") # [250, 720, 1280, 3])

        if os.path.exists(keyframe_path):
            keyframe = np.load(keyframe_path)
            keypoint = self.keypoint_detection(keyframe)
        else:
            raise ValueError(f"keyframe_path={keyframe_path} not exists!")

        fps_ori = info["video_fps"]
        if self.fixed_fps is not None:
            frame_stride_float = frame_stride * (1.0 * fps_ori / self.fixed_fps)
            frame_stride = int(frame_stride_float)

        ## to avoid extreme cases when fixed_fps is used # 
        frame_stride = max(frame_stride, 1)
        
        ## get valid range (adapting case by case)
        frame_num = len(video)
        required_frame_num = math.ceil(2 * fps_ori)
        random_range = frame_num - required_frame_num
        start_idx = random.randint(0, random_range) if random_range > 0 else 0

        # ============== 2. sample keyframes ==========================
        # frame_indices_for_selection, frame_strides, frame_indices= self.select_frame(start_idx, self.video_length, self.fixed_fps, fps_ori, index)



        _, _, key_frame_indices = self.select_keyframe(keypoint, start_idx, self.video_length, self.fixed_fps, fps_ori)

        frame_indices_for_selection, frame_strides, frame_indices = self.select_uniform_frame(start_idx, self.video_length, self.fixed_fps, fps_ori, index)
        frame_strides_fps = torch.tensor(np.array([int(fps_ori / max(fs, 1)) for fs in frame_strides]))
        fps_clip = int(frame_strides_fps.float()[1:].mean())
        frame_indices_for_selection = torch.from_numpy(frame_indices_for_selection).long()


        frames = torch.index_select(video, 0, frame_indices_for_selection)

        is_keyframe = [1 if i in key_frame_indices else 0 for i in frame_indices]
        is_keyframe = torch.tensor(is_keyframe).long()

        if sum(is_keyframe) <= 1:
            is_keyframe[0] = 1
            is_keyframe[-1] = 1

        if waveform.size(0) == 1:
            waveform = waveform.expand(2, -1)
        sr = info["audio_fps"]
       
        start = Fraction(start_idx, frame_num) * frame_num / fps_ori
        end = start + 2.0
        waveform_melspec, waveform_clip = transform_audio_data(waveform, sr, start, end)

        ## ============== 3. Process video data ==========================
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = frames.permute(3, 0, 1, 2) # [t,h,w,c] -> [c,t,h,w]

        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2



        # if self.fps_max is not None and fps_clip > self.fps_max:
        #     fps_clip = self.fps_max

        if waveform_clip.size(-1) >= 33000:
            raise ValueError(f"Too long {waveform_clip.size(-1)}")

        
        data = {'video': frames, 'caption': caption, 'audio': waveform_melspec, 
                'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'frame_strides': frame_strides, 'frame_indices': frame_indices, 'frame_strides_fps': frame_strides_fps, "is_keyframe": is_keyframe, "key_frame_indices": key_frame_indices,
                'audio_waveform': F.pad(waveform_clip,(0, 33000-waveform_clip.size(-1))), 'sr': sr, "length": waveform_clip.size(-1)}

        return data
    
    def __len__(self):
        return len(self.all_video_paths)


import matplotlib
import matplotlib.pyplot as plt

def plot_waveform(waveform, sample_rate, title="Waveform", xlim=None, ylim=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f'Channel {c+1}')
        if xlim:
            axes[c].set_xlim(xlim)
        if ylim:
            axes[c].set_ylim(ylim)
    # figure.suptitle(title)
    plt.title("")
    plt.axis('off')

    # Apply tight layout
    plt.tight_layout(pad=0)  # Set pad=0 for tightest layout

    # plt.show(block=False)

if __name__== "__main__":
    import cv2
    import PIL
    # data_dir = '/dockerx/local/data/VGGSound/scratch/shared/beegfs/hchen/train_data/VGGSound_final/video/'
    # keyframe_dir = '/dockerx/local/data/VGGSound_audio_scores/label'
    # caption_dir = '/dockerx/local/data/VGGSound/vggsound.csv'
    #     keyframe_dir: /dockerx/local/data/VGGSound_audio_scores/label
    #     caption_dir: /dockerx/local/data/VGGSound/vggsound.csv
    data_dir = "/dockerx/local/data/AVSync15/train" ## path to the data directory
    keyframe_dir = "/dockerx/local/data/AVSync15/train_curves_npy" ## path to the keyframe directory 

    L = 12
    dataset = AVSync15(data_dir,
                keyframe_dir=keyframe_dir,  
                 subsample=None,
                 video_length=L,
                 resolution=[320, 512],
                 frame_stride=1,
                 spatial_transform="resize_center_crop",
                 crop_resolution=None,
                 fps_max=None,
                 load_raw_resolution=True,
                 fixed_fps=L//2
                 )
    dataloader = DataLoader(dataset,
                    batch_size=1,
                    # num_workers=4,
                    shuffle=False)

    
    import sys
    sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    from utils.save_video import tensor_to_mp4
    # for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
    from tqdm import tqdm
    save_frame_folder = "save/vis_frames_fps12"
    os.makedirs(save_frame_folder, exist_ok=True)
    for i, batch in enumerate(tqdm(dataloader)):
        video = batch['video'][0]
        audio = batch['audio_waveform'][0][:, :batch['length'][0]]
        sample_rate = batch['sr'][0]
        frame_indices = batch['frame_indices'][0]
        fps_clip = batch['fps'][0]

        # need /: frame index, 12 frames videos, 


        # print(batch['path'][0], batch['video'][0].size(), batch['audio_waveform'][0].size(), batch['length'])

        img_frames = torch.cat([video[:, l] for l in range(L)], dim=-1)
        img_frames = (img_frames * 0.5 + 0.5)
        h, w = img_frames.size(1), img_frames.size(2)
        save_path = 'debug.png'
        cv2.imwrite(save_path, img_frames.permute(1, 2, 0).numpy()[:, :, ::-1] * 255)
        print(frame_indices)
        print(fps_clip)
        
        import ipdb; ipdb.set_trace()

        # plot_waveform(audio.mean(0, keepdim=True), 16000, xlim=(0, 2))
        # plt.savefig(f"waveform.png")

        # audio_img = PIL.Image.open("waveform.png")
        # audio_img = audio_img.convert('RGB').resize((w, audio_img.size[1]), resample=PIL.Image.NEAREST)
        # audio_img = transforms.ToTensor()(audio_img)[:3,]
        
        # # concat the audio image to the video and save
        # img_frames = torch.cat([img_frames, audio_img], dim=1)
        
        # save_name = batch['path'][0].replace('.mp4','keyframe.png')
        # save_path = os.path.join(save_frame_folder, save_name)
        # os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # # print(save_path)
        
        # cv2.imwrite(save_path, img_frames.permute(1, 2, 0).numpy()[:, :, ::-1] * 255)
        # import ipdb; ipdb.set_trace()
        
        # print(batch['path'][0])


        # import ipdb; ipdb.set_trace()
        
        # print(batch['audio_waveform'].size())
        # name = batch['path'][0].split('videos/')[-1].replace('/','_')
        # tensor_to_mp4(video, save_dir+'/'+name, fps=8)

