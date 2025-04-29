import os
from moviepy.editor import VideoFileClip, CompositeVideoClip

def combine_videos_audio(folder1, folder2, folder3):
    for category in os.listdir(folder1):
        category_path1 = os.path.join(folder1, category)
        category_path2 = os.path.join(folder2, category)
        category_path3 = os.path.join(folder3, category)

        if not os.path.exists(category_path3):
            os.makedirs(category_path3)
        
        for video_name in os.listdir(category_path1):
            
            video_path1 = os.path.join(category_path1, video_name)

            if "Scene-" in video_name:
                video_name = video_name[:25]+video_name[35:]
            
            video_path2 = os.path.join(category_path2, video_name[:25]+'_'+video_name[26:])
            video_path3 = os.path.join(category_path3, video_name[:25]+'_'+video_name[26:])
            

            # try:
            video1 = VideoFileClip(video_path1)
            video2 = VideoFileClip(video_path2)

            audio2 = video2.audio
            final_video = video1.set_audio(audio2)

            if video_name.startswith("-"):
                video_path_should_be = os.path.join(category_path3, video_name[1:])
                final_video.write_videofile(video_path_should_be, codec="libx264", audio_codec="aac")
                os.rename(video_path_should_be, video_path3)
            else:
                final_video.write_videofile(video_path3, codec="libx264", audio_codec="aac")
            
            # except:
            #     import ipdb; ipdb.set_trace()
                




folder1 = "/dockerx/local/repo/DynamiCrafter/save/training_512_avsyn_qformer_finetune/samples"
# folder1 = "/dockerx/local/repo/DynamiCrafter/save/training_512_avsyn_qformer_finetune/samples_interpolated"
folder2 = "/dockerx/local/repo/ASVA/checkpoints/audio-cond_animation/avsync15_audio-cond_cfg/evaluations/checkpoint-37000/AG-4.0_TG-1.0/seed-0/videos"
folder3 = "/dockerx/local/repo/DynamiCrafter/save/training_512_avsyn_qformer_finetune/sample_audio_16"

combine_videos_audio(folder1, folder2, folder3)