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


            # try:
            video1 = VideoFileClip(video_path1)
            video2 = video1.subclip(0,2)
            audio2 = video1.audio
            

            audio2 = video2.audio
            final_video = video1.set_audio(audio2)

            final_video.write_videofile(video_path3, codec="libx264", audio_codec="aac")
            
            # except:
            #     import ipdb; ipdb.set_trace()
                




folder1 = "/dockerx/share/DynamiCrafter/save/inference_512_avsyn_24/samples"
folder2 = "/dockerx/share/DynamiCrafter/save/inference_512_avsyn_24/samples"
folder3 = "/dockerx/share/DynamiCrafter/save/inference_512_avsyn_24/samples_12"

combine_videos_audio(folder1, folder2, folder3)