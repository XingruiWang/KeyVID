import os
from moviepy.editor import VideoFileClip, CompositeVideoClip
from moviepy.editor import VideoFileClip, clips_array
def combine_videos_audio(folder1, folder2, folder3):
    for category in os.listdir(folder1):
        category_path1 = os.path.join(folder1, category)
        category_path2 = os.path.join(folder2, category)
        category_path3 = os.path.join(folder3, category)

        if not os.path.exists(category_path3):
            os.makedirs(category_path3)
        
        for video_name in os.listdir(category_path1):
            
            video_path1 = os.path.join(category_path1, video_name)


            
            video_path2 = os.path.join(category_path2, video_name)
            video_path3 = os.path.join(category_path3, video_name)
            

            

            # Load the two video clips
            video1 = VideoFileClip(video_path1)
            video2 = VideoFileClip(video_path2)

            # Resize the second video to match the height of the first one
            video2_resized = video2.resize(height=video1.h)

            # Concatenate the two videos side by side
            final_video = clips_array([[video1, video2_resized]])

            # Set the audio of the final video to be the audio of the first video
            final_video = final_video.set_audio(video1.audio)

            # Export the final concatenated video
            final_video.write_videofile(video_path3, codec="libx264", audio_codec="aac")



ours = "/dockerx/share/DynamiCrafter/save/training_512_avsyn_qformer_12_keyframe_framequeery_60audio_12frames-multi_cond/ASVA"
folder2 = "/dockerx/share/ASVA/checkpoints/audio-cond_animation/avsync15_audio-cond_cfg/evaluations/checkpoint-37000/AG-4.0_TG-1.0/seed-0/videos_test"
folder3 = "/dockerx/share/DynamiCrafter/save/comparision"

if not os.path.exists(folder3):
    os.makedirs(folder3)

combine_videos_audio(ours, folder2, folder3)