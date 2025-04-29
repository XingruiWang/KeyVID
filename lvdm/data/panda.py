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


class Panda(Dataset):
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
                 flip = True
                 ):
        # self.meta_path = meta_path
        self.data_dir = data_dir # AVSync15/train
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
        
        self.caption = pd.read_csv(caption_dir)
        self._load_metadata(self.caption)


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
    
    def _load_metadata(self, caption):
        self.all_video_paths = []
        original_length = 0
        filter_length = 0
        for index, row in caption.iterrows():
            # path,num_frames,height,width,aspect_ratio,fps,resolution,aes,text
            video_name = row.iloc[0]
            cap = row.iloc[-1]
            num_frames = row.iloc[1]
            fps = row.iloc[5]

            if os.path.exists(os.path.join(self.data_dir,video_name)) and num_frames >= 60 and int(fps) == 30:
                self.all_video_paths.append((cap, video_name))
                filter_length += 1
            original_length += 1
            # Perform your desired operation on the first column value
        print(f"Total {filter_length} (before {original_length}) videos loaded.")

    def _get_video_path(self, sample):
        raise NotImplementedError
        rel_video_fp = os.path.join(sample['page_dir'], str(sample['videoid']) + '.mp4')
        full_video_fp = os.path.join(self.data_dir, 'videos', rel_video_fp)
        return full_video_fp
    
    def normalize_label(self, label):
        raise NotImplementedError
        # Normalize label here
        if np.max(label) == np.min(label):
            return label - np.min(label)
        else:
            normalized_label = (label - np.min(label)) / (np.max(label) - np.min(label))
            return normalized_label

    def interpolate_label(self, label, keypoint=None):
        raise NotImplementedError
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
        raise NotImplementedError
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

    def select_frame(self, start_idx, target_length, target_fps=30, fps_ori=30, index=0):
        # Random select 6 keyframes where the keypoint is 1
        end_idx = start_idx + round(2 * fps_ori) # 12 / 6 * 24 = 48

        # rand_num = random.random() Not use random
        if index % 2 == 0:
            random_keypoint = np.random.choice(range(end_idx - start_idx), target_length, replace=False)
            random_keypoint = np.sort(random_keypoint)

        elif index % 2 == 1:
            # uniform select target_length keyframes between start_idx and end_idx
            random_fs = random.randint(1, int(2*fps_ori/target_length))
            # random_fs = 2*fps_ori/target_length
            if random_fs * target_length >= end_idx - start_idx:
                random_fs = (end_idx - start_idx) // target_length
            random_keypoint = np.array([t * random_fs for t in range(target_length)]).astype(float) 
        
        random_keypoint_condition = random_keypoint * 30 / fps_ori 
        random_keypoint_condition = random_keypoint_condition.astype(int)

        
        frame_stride = random_keypoint_condition[1:] - random_keypoint_condition[:-1]
        frame_stride = np.insert(frame_stride, 0, frame_stride[0])
        return random_keypoint.astype(int)+start_idx, frame_stride.astype(float), random_keypoint_condition.astype(int)
    
    # def check_validation(self, index):
    #     frame_stride = self.frame_stride

    #     # index = index % len(self.metadata)
    #     caption, video_name = self.all_video_paths[index]

    #     video_path = os.path.join(self.data_dir, video_name)


    #     # ================== 1. Load videos ===================
    #     if self.load_raw_resolution:
    #         video_reader = VideoReader(video_path, ctx=cpu(0))
    #     else:
    #         video_reader = VideoReader(video_path, ctx=cpu(0), width=530, height=300)
    #     if len(video_reader) < self.video_length:
    #         raise ValueError(f"video length ({len(video_reader)}) is smaller than target length({self.video_length})")

    #     fps_ori = video_reader.get_avg_fps()
        
    #     ## get valid range (adapting case by case)
    #     frame_num = len(video_reader)

    #     return index, video_name, frame_num, fps_ori, caption
       
        
    def __getitem__(self, index):
        frame_stride = self.frame_stride

        # index = index % len(self.metadata)
        caption, video_path = self.all_video_paths[index]

        video_path = os.path.join(self.data_dir, video_path)


        # ================== 1. Load videos ===================
        video, waveform, info = torchvision.io.read_video(video_path, pts_unit="sec") # [250, 720, 1280, 3])

        fps_ori = info["video_fps"]
        
        ## get valid range (adapting case by case)

        frame_num = len(video)
        required_frame_num = math.ceil(2 * fps_ori)+1
        random_range = frame_num - required_frame_num
        start_idx = random.randint(0, random_range) if random_range > 0 else 0
        frame_indices_for_selection, frame_strides, frame_indices= self.select_frame(start_idx, self.video_length, self.fixed_fps, fps_ori, index)

        frame_indices_for_selection = torch.from_numpy(frame_indices_for_selection).long()
        
        # end_idx = frame_indices[-1]
        try:
            frames = torch.index_select(video, 0, frame_indices_for_selection)
        except:
            print(video.shape)
            print(frame_indices_for_selection)
            print(frame_indices)
            print(frame_strides)
            # with open("error.txt", "w") as f:
            #     f.write(f"video path: {video_path}\n")
            #     f.write(f"video shape: {video.shape}\n")
            #     f.write(f"frame_indices_for_selection: {frame_indices_for_selection}\n")
            #     f.write(f"frame_indices: {frame_indices}\n")
            #     f.write(f"frame_strides: {frame_strides}\n")
            # raise NotImplementedError(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices_for_selection)} / {frame_num}]")
            return self.__getitem__(index+1)
 
            # raise NotImplementedError(f"Get frames failed! path = {video_path}; [max_ind vs frame_total:{max(frame_indices)} / {frame_num}]")
        
        # =============== 2. Load and transform audio data ==========================
        # video_, waveform, info = torchvision.io.read_video(video_path, pts_unit="sec") # [250, 720, 1280, 3])
        if waveform.size(0) == 1:
            waveform = waveform.expand(2, -1)
        try:
            sr = info["audio_fps"]
            start = Fraction(start_idx, frame_num) * frame_num / fps_ori
            end = start + 2.0
            waveform_melspec, waveform_clip = transform_audio_data(waveform, sr, start, end)
        except:
            sr = 16000
            waveform_melspec = torch.zeros(1, 128, 204)
            waveform_clip = torch.zeros(2, 32000)
        ## ============== 3. Process video data ==========================
        assert(frames.shape[0] == self.video_length),f'{len(frames)}, self.video_length={self.video_length}'
        frames = frames.permute(3, 0, 1, 2) # [t,h,w,c] -> [c,t,h,w]

        if self.spatial_transform is not None:
            frames = self.spatial_transform(frames)
        
        if self.resolution is not None:
            assert (frames.shape[2], frames.shape[3]) == (self.resolution[0], self.resolution[1]), f'frames={frames.shape}, self.resolution={self.resolution}'
        
        ## turn frames tensors to [-1,1]
        frames = (frames / 255 - 0.5) * 2
        
        # if self.flip and random.random() > 0.5:
        #     frames = torch.flip(frames, dims=(3,))
            
        
        fps_clip = fps_ori // frame_stride
        if self.fps_max is not None and fps_clip > self.fps_max:
            fps_clip = self.fps_max

        if waveform_clip.size(-1) >= 33000:
            raise ValueError(f"Too long {waveform_clip.size(-1)}")
        # print(f"fps_clip={fps_clip}, frame_stride={frame_stride}, waveform_clip={waveform_clip.size(-1)}")
        
        data = {'video': frames, 'caption': caption, 'audio': waveform_melspec, 
                'path': video_path, 'fps': fps_clip, 'frame_stride': frame_stride, 'frame_strides': frame_strides, 'frame_indices': frame_indices,
                'audio_waveform': F.pad(waveform_clip,(0, 33000-waveform_clip.size(-1))), 'sr': sr, "length": waveform_clip.size(-1)}
                # 'audio_waveform': waveform_clip, 'sr': sr, "length": waveform_clip.size(-1)}
        # print("Start: ", start, "End: ", end, "Length: ", waveform_clip.size(-1))
        # print("Frames selected", frame_indices)
        # print("Frames selected relative", [f / fps_ori for f in frame_indices])
        # import ipdb; ipdb.set_trace()
        
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
    data_dir = '/dockerx/local/data/panda_aes/clips'
    caption_dir = '/dockerx/local/data/panda_aes/meta_caption_info_headless.csv'
    #     keyframe_dir: /dockerx/local/data/VGGSound_audio_scores/label
    #     caption_dir: /dockerx/local/data/VGGSound/vggsound.csv
    # data_dir = "/dockerx/local/repo/DynamiCrafter/data/AVSync15/train" ## path to the data directory

    L = 12
    dataset = Panda(data_dir,
                 caption_dir=caption_dir,
                 subsample=None,
                 video_length=L,
                 resolution=[256,448],
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

    metadata = []

    


    
    # import sys
    # sys.path.insert(1, os.path.join(sys.path[0], '..', '..'))
    # from utils.save_video import tensor_to_mp4
    # # for i, batch in tqdm(enumerate(dataloader), desc="Data Batch"):
    # from tqdm import tqdm
    # save_frame_folder = "save/vis_frames_fps12"
    # os.makedirs(save_frame_folder, exist_ok=True)
    for i, batch in enumerate(tqdm(dataloader)):
        video = batch['video'][0]
        audio = batch['audio_waveform'][0][:, :batch['length'][0]]
        sample_rate = batch['sr'][0]

        print(batch['path'][0], batch['video'][0].size(), batch['audio_waveform'][0].size(), batch['length'])
        print(batch['frame_indices'][0], batch['frame_strides'][0])

    #     # img_frames = torch.cat([video[:, l] for l in range(L)], dim=-1)
    #     # img_frames = (img_frames * 0.5 + 0.5)
    #     # h, w = img_frames.size(1), img_frames.size(2)
        
    #     # plot_waveform(audio.mean(0, keepdim=True), 16000, xlim=(0, 2))
    #     # plt.savefig(f"waveform.png")

    #     # audio_img = PIL.Image.open("waveform.png")
    #     # audio_img = audio_img.convert('RGB').resize((w, audio_img.size[1]), resample=PIL.Image.NEAREST)
    #     # audio_img = transforms.ToTensor()(audio_img)[:3,]
        
    #     # # concat the audio image to the video and save
    #     # img_frames = torch.cat([img_frames, audio_img], dim=1)
        
    #     # save_name = batch['path'][0].replace('.mp4','keyframe.png')
    #     # save_path = os.path.join(save_frame_folder, save_name)
    #     # os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     # # print(save_path)
        
    #     # cv2.imwrite(save_path, img_frames.permute(1, 2, 0).numpy()[:, :, ::-1] * 255)
    #     # import ipdb; ipdb.set_trace()
        
    #     # print(batch['path'][0])


    #     # import ipdb; ipdb.set_trace()
        
    #     # print(batch['audio_waveform'].size())
    #     # name = batch['path'][0].split('videos/')[-1].replace('/','_')
    #     # tensor_to_mp4(video, save_dir+'/'+name, fps=8)

