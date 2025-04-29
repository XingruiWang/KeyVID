import os 
import cv2
import sys
import numpy as np
import torchvision
from tqdm import tqdm
sys.path.append(".")


def frames48(videos):
    # video shape: f, h, w, c
    f, h, w, c = videos.shape
    assert 48 % f == 0
    if f == 48:
        return videos

    repeat = 48 // f
    videos = videos.repeat(repeat, 1, 1, 1)

    return videos

# def frames12(videos):
#     # video shape: f, h, w, c
#     f, h, w, c = videos.shape
#     assert 48 % f == 0
#     if f == 48:
#         return videos

#     repeat = 48 // f
#     videos = videos.repeat(repeat, 1, 1, 1)

#     return videos

def concat_results(video_dirs = []):

    '''
    video_folder
    |__ category1
    |   |__ video1.mp4
    |   |__ video2.mp4
    |__ category2


    '''
    pivot_folder = video_dirs[0]
    
    all_path = []
    for categorie in os.listdir(pivot_folder):
        for video in os.listdir(os.path.join(pivot_folder, categorie)):
            all_path.append(os.path.join(categorie, video))
    
    for sample_path in tqdm(all_path):
        sample = []
        for video_dir in video_dirs:

            try:
                video_path = os.path.join(video_dir, sample_path)
                video, waveform, info = torchvision.io.read_video(video_path, pts_unit="sec") # [12, 320, 512, 3])
            except:
                video_path = os.path.join(video_dir, sample_path.replace('clip-', 'clip-0'))
                video, waveform, info = torchvision.io.read_video(video_path, pts_unit="sec") # [12, 320, 512, 3])
                

            sample.append(video)
        
        # sample: [video1, video2, video3]
        # video1: [f, h, w, c]
        videos = []
        for video in sample:
            videos.append(video)
            # videos.append(frames48(video))
        
        # concat
        video = np.concatenate(videos, axis=2)
        # save
        
        save_path =  f'/dockerx/local/tmp/visualization/{sample_path}'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        torchvision.io.write_video(save_path, video, 24, audio_array=waveform, audio_fps=16000, audio_codec='aac')


    
# gt_path = '/dockerx/groups/ASVA/datasets/AVSync15/videos'
# generate_path = '/dockerx/local/tmp/asva_12_kf/samples'
# # path_1 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_keyframe/epoch=479-step=11520-2_audio_7.5_img_2.0/samples'
# # path_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_keyframe_add_idx/epoch=219-step=2640_audio_7.5_img_2.0/samples'
# # path_3 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_keyframe_no_idx/epoch=419-step=5040_audio_7.5_img_2.0/samples' 

# # path_1 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_keyframe/epoch=479-step=11520-3_audio_7.5_img_2.0/samples'
# path_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_audio_7.5_img_2.0/samples'
# # path_3 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_keyframe_no_idx/epoch=419-step=5040-2_audio_7.5_img_2.0/samples'
# # path_4 = "/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_kf_no_idx/epoch=849-step=10200-kf_audio_7.5_img_2.0/samples"

# # concat_results([gt_path, generate_path])

# gt_path = '/dockerx/groups/ASVA/datasets/AVSync15/videos'

folder_1 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_kf_interp-948e/epoch=969-step=11640_audio_4.0_img_2.0_inpainting_step_0/ASVA'
folder_2 = '/dockerx/share/Dynamicrafter_audio/save/asva/asva_12_kf_interp-948e/epoch=969-step=11640-uniform_audio_4.0_img_2.0_inpainting_step_0/ASVA'


folder_3 = '/dockerx/local/repo/ASVA/checkpoints/audio-cond_animation/avsync15_audio-cond_cfg/evaluations/checkpoint-37000/AG-4.0_TG-1.0/seed-0/videos'

concat_results([folder_3])