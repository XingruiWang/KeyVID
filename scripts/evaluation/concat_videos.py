# Concat video for visualization


path = '/dockerx/local/tmp/tmp/selected'


# read video and save image frames and plot the audio

import os


import numpy as np
import torch
import torchvision
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import librosa.display
import cv2

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

for video_name in tqdm(os.listdir(path)):
    video_path = os.path.join(path, video_name)
    video, waveform, info = torchvision.io.read_video(video_path, pts_unit="sec") # [12, 320, 512, 3])
    video = video.numpy()

    os.makedirs(f'/dockerx/local/tmp/selected_frames/{video_name[:-4]}', exist_ok=True)
    
    cat_frames = []
    cat_frames_2 = []
    for i in range(video.shape[0]):
        frame = video[i]
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        # frame.save(f'/dockerx/local/tmp/selected_frames/{video_name[:-4]}/{i:03d}.png')

        if i % 1 == 0:
            # import ipdb; ipdb.set_trace()
            frame_vis = np.array(frame)
            empty = np.zeros((256, 420, 3))

            empty[:, 82:82+256, :] = frame_vis

            cat_frames.append(empty)
            # cat_frames.append(np.array(frame)[:, :512, :])
            # cat_frames_2.append(np.array(frame)[:, 512:, :])

    
    cat_frames = np.concatenate(cat_frames, axis=1)
    # cat_frames_2 = np.concatenate(cat_frames_2, axis=1)

    cat_frames = Image.fromarray(cat_frames.astype(np.uint8))
    cat_frames.save(f'/dockerx/local/tmp/selected_frames/{video_name[:-4]}/cat_frames_kf.png')

    # cat_frames.save(f'/dockerx/local/tmp/selected_frames/{video_name[:-4]}/cat_frames_asva.png')

    # cat_frames_2 = Image.fromarray(cat_frames_2)
    # cat_frames_2.save(f'/dockerx/local/tmp/selected_frames/{video_name[:-4]}/cat_frames_uniform.png')
    
    # plot audio

    # plot_waveform(waveform, 16000)
    # plt.savefig(f'/dockerx/local/tmp/selected_frames/{video_name[:-4]}/audio.png')
    # plt.close()


    