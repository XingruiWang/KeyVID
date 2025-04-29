import cv2
import numpy as np
import os
# Load the video
from tqdm import tqdm

input_root = '/dockerx/share/DynamiCrafter/save/inference_512_avsyn_24/samples'
# video_path = 'lions_roaring/3yXCtpvjz6E_000017_000027-1.0_4.5_clip-00.mp4'

def interpolation(input_root, output_root, cate, video_name):
    video_path = os.path.join(input_root, cate, video_name)
    cap = cv2.VideoCapture(video_path)

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    # Calculate the interpolation ratio
    original_frame_count = len(frames)
    new_frame_count = 12
    interpolation_ratio = original_frame_count / new_frame_count

    # Create the new frame list
    new_frames = []

    for i in range(new_frame_count):
        original_index = i * interpolation_ratio
        lower_index = int(np.floor(original_index))
        upper_index = int(np.ceil(original_index))

        if lower_index == upper_index:
            new_frames.append(frames[lower_index])
        else:
            alpha = original_index - lower_index
            new_frame = cv2.addWeighted(frames[lower_index], 1 - alpha, frames[upper_index], alpha, 0)
            new_frames.append(new_frame)


    # Set up the video writer for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps=6
    output_path = os.path.join(output_root, cate, video_name)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in new_frames:
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()

    return output_path

output_root = '/dockerx/share/DynamiCrafter/save/inference_512_avsyn_24/samples_interpolated/'


for cate in tqdm(os.listdir(input_root)):
    for video_name in os.listdir(os.path.join(input_root, cate)):
        interpolation(input_root, output_root, cate, video_name)