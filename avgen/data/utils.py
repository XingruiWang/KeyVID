from typing import List, Union, Literal, Tuple
import itertools
import PIL
from einops import rearrange

import numpy as np
import torch
import torchaudio
import torchvision
from torchvision.io import VideoReader
torchvision.set_video_backend("video_reader")
import torchvision.transforms as transforms

from transformers import ImageProcessingMixin

from imagebind.data import waveform2melspec
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
import random

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711])

VIDEO_SAMPLING_TYPES = Literal["random", "center"]
AUDIO_SAMPLING_TYPES = Literal["random", "center"]


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
        waveform_clip, sample_rate, num_mel_bins, target_length
    )  # (1, n_mel, n_frame)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    audio_clip = normalize(waveform_melspec)
    
    return audio_clip  # (1, freq, time)


class AudioMelspectrogramExtractor(ImageProcessingMixin):
    
    def __init__(
        self,
        num_mel_bins=128,
        target_length=204,
        sample_rate=16000,
        clip_duration=2,
        mean=-4.268,
        std=9.138
    ):
        super().__init__()
        self.num_mel_bins = num_mel_bins
        self.target_length = target_length
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.mean = mean
        self.std = std
    
    @property
    def max_length_s(self) -> int:
        return self.clip_duration
    
    @property
    def sampling_rate(self) -> int:
        return self.sample_rate
    
    def __call__(
            self,
            waveforms: Union[
                np.ndarray,
                torch.Tensor,
                List[np.ndarray],
                List[torch.Tensor]
            ]
    ):
        if isinstance(waveforms, (np.ndarray, torch.Tensor)) and waveforms.ndim == 2:
            waveforms = [waveforms, ]
        features = []
        
        for waveform in waveforms:
            feature = waveform_to_melspectrogram(
                waveform=waveform,
                num_mel_bins=self.num_mel_bins,
                target_length=self.target_length,
                sample_rate=self.sample_rate,
                clip_duration=self.clip_duration,
                mean=self.mean,
                std=self.std
            )
            features.append(feature)
        features = torch.stack(features, dim=0)
        
        return features # (b c n t)


def load_and_transform_images_stable_diffusion(
        images: Union[List[np.ndarray], torch.Tensor, np.ndarray],
        size=512,
        flip=False,
        randcrop=False,
        normalize=True
):
    """
    @images: (List of) np.uint8 images of shape (h, w, 3)
            or tensor of shape (b, c, h, w) in [0., 1.0]

    """
    
    assert isinstance(images, (List, torch.Tensor, np.ndarray)), type(images)
    if isinstance(images, List):
        assert isinstance(images[0], np.ndarray)
        assert images[0].dtype == np.uint8
        assert images[0].shape[2] == 3
        
        # convert np images into torch float tensor
        images = torch.from_numpy(
            rearrange(np.stack(images, axis=0), "f h w c -> f c h w")
        ).float() / 255.
    elif isinstance(images, np.ndarray):
        assert isinstance(images, np.ndarray)
        assert images.dtype == np.uint8
        assert images.shape[3] == 3
        
        # convert np images into torch float tensor
        images = torch.from_numpy(
            rearrange(images, "f h w c -> f c h w")
        ).float() / 255.
        
    assert images.shape[1] == 3
    assert torch.all(images<= 1.0) and torch.all(images >= 0.0)
    
    h, w = images.shape[-2:]
    if isinstance(size, int):
        target_h, target_w = size, size
    else:
        target_h, target_w = size
    
    # first crop the image
    target_aspect_ratio = float(target_h) / target_w
    curr_aspect_ratio = float(h) / w
    if target_aspect_ratio >= curr_aspect_ratio: # trim w
        trimmed_w = int(h / target_aspect_ratio)
        images = images[:, :, :, (w-trimmed_w)//2: (w-trimmed_w)//2+trimmed_w]
    else: # trim h
        trimmed_h = int(w * target_aspect_ratio)
        images = images[:, :, (h - trimmed_h) // 2: (h - trimmed_h) // 2 + trimmed_h]
    
    transform_list = [
        transforms.Resize(
            size,
            interpolation=transforms.InterpolationMode.BILINEAR,
            antialias=True
        ),
    ]
    
    # assert not randcrop
    if randcrop:
        transform_list.append(transforms.RandomCrop(size))
    else:
        transform_list.append(transforms.CenterCrop(size))
        
    if flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=1.0))

    if normalize:
        transform_list.append(transforms.Normalize([0.5], [0.5]))
    
    data_transform = transforms.Compose(transform_list)
    
    images = data_transform(images)
    return images

def load_video_clip_from_videoreader(
        av_reader,
        clip_start_timestamp,
        clip_duration,
        video_fps,
        video_num_frame,
        image_size,
        flip=False,
        randcrop=False,
        normalize=False,
        motion_score=None,
        keyframe_select_fn=None
):
    av_reader.set_current_stream("video")
    keyframe_coverage = 1. / video_fps
    
    video_frames = []
    frame_timestamp = clip_start_timestamp
    for i, frame in enumerate(itertools.takewhile(
            lambda x: x['pts'] <= clip_start_timestamp + clip_duration + keyframe_coverage / 2.,
            av_reader.seek(max(clip_start_timestamp, 0.))
    )):
        if frame["pts"] >= frame_timestamp:
            video_frames.append(frame["data"])  # (c, h, w) tensor [0, 255]
            frame_timestamp += keyframe_coverage
        
        if len(video_frames) == video_num_frame:
            break
    
    if len(video_frames) < video_num_frame:
        res_length = video_num_frame - len(video_frames)
        for _ in range(res_length):
            video_frames.append(video_frames[-1])


    
    video_frames = torch.stack(video_frames, dim=0).float() / 255.
    
    video_frames = load_and_transform_images_stable_diffusion(
        video_frames,
        size=image_size,
        flip=flip,
        randcrop=randcrop,
        normalize=normalize
    ).float()  # (n_frame, 3, h, w) in range [0., 1.]
    
    return video_frames


def keypoint_detection(label, prominence=0.1):
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

def select_keyframe(keyframe, start_idx, target_length, target_fps, fps_ori):
    # print(keyframe, start_idx, target_length, target_fps, fps_ori)

    # 1. Random select 6 keyframes where the keypoint is 1
    # 2. Random select 1 keyframes where keypoint is -1 between each two keyframes
    # 3. Random select until the target length is reached for keypoint 0


    end_idx = start_idx + round(target_length / target_fps * fps_ori)

    keyframe = keyframe[start_idx:end_idx]


    # print(f"start_idx: {start_idx}, end_idx: {end_idx}, target_length: {target_length}, target_fps: {target_fps}, fps_ori: {fps_ori}")

    pos_1 = np.where(keyframe == 1)[0]
    pos_neg_1 = np.where(keyframe == -1)[0]


    if len(pos_1) == 0:
        random_keypoint = [0, end_idx - start_idx - round(fps_ori / target_fps)]
    else:
        select_num = min(6, len(pos_1))
        random_peak = np.random.choice(pos_1, select_num, replace=False)
        random_valley = np.array([])
        if 0 not in random_peak:
            random_peak = np.insert(random_peak, 0, 0)
        else:
            select_num -= 1
        random_peak = np.sort(random_peak)
        for i in range(0, select_num):
            peak_1, peak_2 = random_peak[i], random_peak[i+1]

            all_valley = pos_neg_1[(pos_neg_1 > peak_1) & (pos_neg_1 < peak_2)]
            if len(all_valley) > 0:
                random_v= np.random.choice(all_valley, 1)
                random_valley = np.concatenate([random_valley, random_v])
        
        random_keypoint = np.sort(np.concatenate([random_peak, random_valley]))
    
    if len(random_keypoint) < target_length:
        idx_to_add = np.array([])
        num_padding = target_length - len(random_keypoint)

        start, intervel_len = random_keypoint[:-1], [random_keypoint[i+1] - random_keypoint[i] for i in range(len(random_keypoint)-1)]

        
        # start = np.insert(start, -1, random_keypoint[-1])  
        # intervel_len = np.insert(intervel_len, -1, end_idx - random_keypoint[-1])  

        # num_padding_interval = intervel_len * num_padding / sum(intervel_len)
        # num_padding_interval = [num_padding * l / sum(intervel_len) for l in intervel_len]
        # import ipdb; ipdb.set_trace()
        
        for i in range(len(intervel_len)):
            if i == len(intervel_len) - 1 and num_padding > 0:
                idx_to_add = np.concatenate([idx_to_add, np.linspace(random_keypoint[i], random_keypoint[i+1], num_padding+1, endpoint=False).astype(int)[1:]])
            else:

                weight = intervel_len[i] / sum(intervel_len)

                padd_num = min(num_padding, int(weight // 1) + int(random.random() < weight % 1) )
                if padd_num > 0:
                    num_padding -= padd_num
                    
                    idx_to_add = np.concatenate([idx_to_add, np.linspace(random_keypoint[i], random_keypoint[i+1], padd_num+1, endpoint=False).astype(int)[1:]])
        
        random_keypoint = np.sort(np.concatenate([random_keypoint, idx_to_add]))
    random_keypoint = np.unique(random_keypoint)
    if len(random_keypoint) > target_length:
        random_keypoint = random_keypoint[:target_length]
    elif len(random_keypoint) < target_length:
        unselected = set(range(end_idx - start_idx)) - set(random_keypoint)
        extra = np.random.choice(list(unselected), target_length - len(random_keypoint), replace=False)
        random_keypoint = np.sort(np.concatenate([random_keypoint, extra]))
    
    frame_stride = random_keypoint[1:] - random_keypoint[:-1]
    frame_stride = np.insert(frame_stride, 0, round(fps_ori / target_fps))

    
    
    random_keypoint_condition = random_keypoint * 24 / fps_ori 
    random_keypoint_condition = np.round(random_keypoint_condition)

    return random_keypoint.astype(int), frame_stride, random_keypoint_condition

def load_video_keyframe_from_videoreader(
        av_reader,
        clip_start_timestamp,
        clip_duration,
        video_fps,
        video_num_frame,
        image_size,
        flip=False,
        randcrop=False,
        normalize=False,
        motion_score=None,
        saved_frame_idx=None):
    av_reader.set_current_stream("video")
    keyframe_coverage = 1. / video_fps
    
    video_frames = []
    frame_timestamp = clip_start_timestamp
    clip_end_timestamp = clip_start_timestamp + clip_duration
    frame_index = []
    # import ipdb; ipdb.set_trace()
    
    for i, frame in enumerate(av_reader.seek(0)):
        if frame["pts"] >= clip_start_timestamp and frame["pts"] < clip_end_timestamp:
            frame_index.append(i)
    
    keypoint = keypoint_detection(motion_score, prominence=0.1)
    keyframe, frame_stride, keyframe_cond = select_keyframe(keypoint, frame_index[0], video_num_frame, video_fps, len(frame_index)/clip_duration)
    
    # import ipdb; ipdb.set_trace()
    assert len(keyframe) == video_num_frame, len(list(set(keyframe))) == len(keyframe)
    
    unselected = []
    selected = []
    for i, frame in enumerate(av_reader.seek(0)):
        if i-frame_index[0] in keyframe:
            video_frames.append(frame["data"])
            selected.append(i)
        else:
            unselected.append(i)

    if len(video_frames) < video_num_frame:
        print(f"keyframe: {keyframe}")
        print(f"selected: {selected}")
        print(f"unselected: {unselected}")
        import ipdb; ipdb.set_trace()
        

    video_frames = torch.stack(video_frames, dim=0).float() / 255.
    
    video_frames = load_and_transform_images_stable_diffusion(
        video_frames,
        size=image_size,
        flip=flip,
        randcrop=randcrop,
        normalize=normalize
    ).float()  # (n_frame, 3, h, w) in range [0., 1.]
    assert video_frames.shape[0] == video_num_frame, video_frames.shape
    return {'video': video_frames, 'keyframe': keyframe_cond, 'frame_stride': frame_stride}


def load_audio_clip_from_videoreader(
        av_reader,
        clip_start_timestamp,
        clip_duration,
        audio_sr,
        load_audio_as_melspectrogram,
        target_audio_sr=16000
):
    av_reader.set_current_stream("audio")
    
    audio_frames = []
    for frame in itertools.takewhile(
            lambda x: x['pts'] <= clip_start_timestamp + clip_duration,
            av_reader.seek(clip_start_timestamp)
    ):
        if frame['pts'] >= clip_start_timestamp and \
                frame['pts'] <= clip_start_timestamp + clip_duration:
            frame_data = frame["data"]
            t, c = frame_data.shape
            frame_data = frame_data.contiguous().view(c, t).contiguous()
            audio_frames.append(frame_data)  # (c, t)
    
    audio = torchaudio.functional.resample(
        torch.cat(audio_frames, dim=1),
        orig_freq=audio_sr,
        new_freq=target_audio_sr
    )  # (C, T)
    
    if load_audio_as_melspectrogram:
        audio = waveform_to_melspectrogram(audio)  # (1, n, t)
    
    return audio
def load_v_clips_uniformly(
        video_path: str,
        video_fps: int = 6,
        video_num_frame: int = 12,
        image_size: Union[int, Tuple[int, int]] = 512,
        num_clips: int = 1,
        load_audio_as_melspectrogram: bool = True,
):
    '''
    Return:
        video_frames: (b f c h w) in [0, 1]
         audio_frames:
            if load_audio_as_melspectrogram is True: (b 1 n t)
            else: List of tensors (b c ti), ti can be different
    '''

    clip_duration = video_num_frame / video_fps
    av_reader = VideoReader(video_path, stream="video")
    meta_data = av_reader.get_metadata()
    video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
    av_duration =video_duration
    # assert av_duration >= clip_duration, [video_path, video_duration, audio_duration]
    
    # 1. Sample clip start times
    if num_clips == 1:
        clip_start_timestamps = np.array([(av_duration - clip_duration) / 2.])
    else:
        clip_start_timestamps = np.linspace(0., av_duration - clip_duration, endpoint=True, num=num_clips)
    
    video_frames = []
    audio_frames = []
    for clip_start_timestamp in clip_start_timestamps:
        video_frames.append(
            load_video_clip_from_videoreader(
                av_reader,
                clip_start_timestamp,
                clip_duration,
                video_fps,
                video_num_frame,
                image_size,
                flip=False,
                randcrop=False,
                normalize=False
            )
        )
      
    
    video_frames = torch.stack(video_frames)  # (b, t, c, h, w)

    return video_frames, None

def load_av_clips_uniformly(
        video_path: str,
        video_fps: int = 6,
        video_num_frame: int = 12,
        image_size: Union[int, Tuple[int, int]] = 512,
        num_clips: int = 1,
        load_audio_as_melspectrogram: bool = True,
):
    '''
    Return:
        video_frames: (b f c h w) in [0, 1]
         audio_frames:
            if load_audio_as_melspectrogram is True: (b 1 n t)
            else: List of tensors (b c ti), ti can be different
    '''

    clip_duration = video_num_frame / video_fps
    av_reader = VideoReader(video_path, stream="video")
    meta_data = av_reader.get_metadata()
    video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
    audio_duration, audio_sr = float(meta_data["audio"]["duration"][0]), int(meta_data["audio"]["framerate"][0])
    av_duration = min(video_duration, audio_duration)
    # assert av_duration >= clip_duration, [video_path, video_duration, audio_duration]
    
    # 1. Sample clip start times
    if num_clips == 1:
        clip_start_timestamps = np.array([(av_duration - clip_duration) / 2.])
    else:
        clip_start_timestamps = np.linspace(0., av_duration - clip_duration, endpoint=True, num=num_clips)
    
    video_frames = []
    audio_frames = []
    for clip_start_timestamp in clip_start_timestamps:
        video_frames.append(
            load_video_clip_from_videoreader(
                av_reader,
                clip_start_timestamp,
                clip_duration,
                video_fps,
                video_num_frame,
                image_size,
                flip=False,
                randcrop=False,
                normalize=False
            )
        )
        audio_frames.append(
            load_audio_clip_from_videoreader(
                av_reader,
                clip_start_timestamp,
                clip_duration,
                audio_sr,
                load_audio_as_melspectrogram
            )
        )
    
    video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
    if load_audio_as_melspectrogram:
        audio_frames = torch.stack(audio_frames)  # (b, 1, c, t)
    
    return video_frames, audio_frames


def load_av_clips_keyframe(
        video_path: str,
        video_fps: int = 6,
        video_num_frame: int = 12,
        image_size: Union[int, Tuple[int, int]] = 512,
        num_clips: int = 1,
        motion_score=None,
        load_audio_as_melspectrogram: bool = True,
        saved_frame_idx=None
):
    '''
    Return:
        video_frames: (b f c h w) in [0, 1]
         audio_frames:
            if load_audio_as_melspectrogram is True: (b 1 n t)
            else: List of tensors (b c ti), ti can be different
    '''

    clip_duration = video_num_frame / video_fps
    av_reader = VideoReader(video_path, stream="video")
    meta_data = av_reader.get_metadata()
    video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
    audio_duration, audio_sr = float(meta_data["audio"]["duration"][0]), int(meta_data["audio"]["framerate"][0])
    av_duration = min(video_duration, audio_duration)
    # assert av_duration >= clip_duration, [video_path, video_duration, audio_duration]
    
    # 1. Sample clip start times
    if num_clips == 1:
        clip_start_timestamps = np.array([(av_duration - clip_duration) / 2.])
    else:
        clip_start_timestamps = np.linspace(0., av_duration - clip_duration, endpoint=True, num=num_clips)
    
    video_frames = []
    audio_frames = []
    keyframes = []
    frame_strides = []

    for j, clip_start_timestamp in enumerate(clip_start_timestamps):
        if saved_frame_idx is not None:
            saved_frame_idx_j = saved_frame_idx[j]
        else:
            saved_frame_idx_j = None
        keyframe = load_video_keyframe_from_videoreader(
                av_reader,
                clip_start_timestamp,
                clip_duration,
                video_fps,
                video_num_frame,
                image_size,
                flip=False,
                randcrop=False,
                normalize=False,
                motion_score=motion_score,
                saved_frame_idx=saved_frame_idx_j
            )
        video_frames.append(keyframe['video'])
        keyframes.append(keyframe['keyframe'])
        frame_strides.append(keyframe['frame_stride'])
        audio_frames.append(
            load_audio_clip_from_videoreader(
                av_reader,
                clip_start_timestamp,
                clip_duration,
                audio_sr,
                load_audio_as_melspectrogram
            )
        )
    
    video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
    if load_audio_as_melspectrogram:
        audio_frames = torch.stack(audio_frames)  # (b, 1, c, t)
    keyframes = torch.tensor(keyframes)
    frame_strides = torch.tensor(frame_strides)
    # import ipdb; ipdb.set_trace()
    
    return video_frames, audio_frames, keyframes, frame_strides


def load_video_clips_uniformly(
        video_path: str,
        video_fps: int = 6,
        video_num_frame: int = 12,
        image_size: Union[int, Tuple[int, int]] = 512,
        num_clips: int = 1
):
    '''
    Return:
        video_frames: (b f c h w) in [0, 1]
    '''
    clip_duration = video_num_frame / video_fps
    av_reader = VideoReader(video_path, stream="video")
    meta_data = av_reader.get_metadata()
    video_duration, orig_video_fps = float(meta_data["video"]["duration"][0]), float(meta_data["video"]["fps"][0])
    
    # 1. Sample clip start times
    if num_clips == 1:
        clip_start_timestamps = np.array([(video_duration - clip_duration) / 2.])
    else:
        clip_start_timestamps = np.linspace(0., video_duration - clip_duration, endpoint=True, num=num_clips)
    
    video_frames = []
    for clip_start_timestamp in clip_start_timestamps:
        video_frames.append(
            load_video_clip_from_videoreader(
                av_reader,
                clip_start_timestamp,
                clip_duration,
                video_fps,
                video_num_frame,
                image_size
            )
        )
    
    video_frames = torch.stack(video_frames)  # (b, t, c, h, w)
    
    return video_frames


def load_image(image_path, image_size = (256, 256)):
    '''
    Return:
        image: tensor (3, h, w) in [0, 1]
    '''
    image = PIL.Image.open(image_path).convert('RGB')
    image = torch.from_numpy(np.array(image))
    image = rearrange(image, "h w c -> 1 c h w") / 255.
    
    image = load_and_transform_images_stable_diffusion(
        image, size=image_size, flip=False, randcrop=False, normalize=False
    )[0].contiguous()
    
    return image


def load_audio_clips_uniformly(
        audio_path: str,
        clip_duration: float = 2.0,
        num_clips: int = 1,
        load_audio_as_melspectrogram: bool = True
):
    '''
    Return:
        audio_frames:
            if load_audio_as_melspectrogram is True: (b 1 n t)
            else: List of b tensors (c t)
    '''
    audio, sr = torchaudio.load(audio_path)
    audio_duration = audio.shape[1] / float(sr)
    
    audio = torchaudio.functional.resample(audio, orig_freq=sr, new_freq=16000)
    
    # 1. Sample clip start times
    if num_clips == 1:
        clip_start_timestamps = np.array([(audio_duration - clip_duration) / 2.])
    else:
        clip_start_timestamps = np.linspace(0., audio_duration - clip_duration, endpoint=True, num=num_clips)
    
    audio_frames = []
    for clip_start_timestamp in clip_start_timestamps:
        
        audio_clip = audio[:, int(clip_start_timestamp*16000):int((clip_start_timestamp +clip_duration) *16000)].contiguous()
        if load_audio_as_melspectrogram:
            audio_clip = waveform_to_melspectrogram(audio_clip)
        audio_frames.append(audio_clip)

    if load_audio_as_melspectrogram:
        audio_frames = torch.stack(audio_frames)  # (b, 1, c, t)
    
    return audio_frames


def get_avsync15_evaluation_data():
    # import ipdb; ipdb.set_trace()
    
    dataset_root = f"./datasets/AVSync15"
    video_root = f"{dataset_root}/videos"
    with open(f"{dataset_root}/test.txt", "r") as f:
    # with open(f"{dataset_root}/sub_train.txt", "r") as f:
        video_paths = [file.strip() for file in f.readlines()]
        categories = [file.split('/')[0] for file in video_paths]
    
    return video_root, video_paths, categories


def get_thegreatesthits_evaluation_data():
    dataset_root = f"./datasets/TheGreatestHits"
    video_root = f"{dataset_root}/videos"
    
    with open(f"{dataset_root}/test.txt", "r") as f:
        video_paths = [file.strip() for file in f.readlines()]
    categories = ["hitting with a stick"] * len(video_paths)
    
    return video_root, video_paths, categories


def get_landscapes_evaluation_data():
    dataset_root = f"./datasets/Landscapes"
    video_root = f"{dataset_root}/videos/test"
    
    with open(f"{dataset_root}/test.txt", "r") as f:
        video_paths = [file.strip() for file in f.readlines()]
    categories = [file.split('/')[0] for file in video_paths]
    
    return video_root, video_paths, categories


def get_evaluation_data(dataset):
    video_path_type = "video"
    
    if dataset == "AVSync15":
        video_root, video_paths, categories = get_avsync15_evaluation_data()
    elif dataset == "TheGreatestHits":
        video_root, video_paths, categories = get_thegreatesthits_evaluation_data()
    elif dataset == "Landscapes":
        video_root, video_paths, categories = get_landscapes_evaluation_data()
    else:
        raise Exception()
    
    return video_root, video_paths, categories, video_path_type


