import argparse
import os
import cv2
import json




def video_to_frames(input_path, output_folder):
    """Extract frames from a video file and save them as images in a folder.

    Args:
        input_path (str): Path to the video file.
        output_folder (str): Path to the folder where the frames will be saved.
    """
    # Create the output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(input_path)

    # Get the total number of frames in the video
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Get the frame rate of the video
    frame_rate = int(video.get(cv2.CAP_PROP_FPS))

    # Get the width and height of the video frames
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read frames from the video and save them as images
    for i in range(total_frames):
        ret, frame = video.read()
        if not ret:
            break

        # Save the frame as an image
        frame_path = os.path.join(output_folder, f"frame_{i:04d}.jpg")
        cv2.imwrite(frame_path, frame)

    # Release the video object
    video.release()

    # save video metadata
    metadata = {
        "total_frames": total_frames,
        "frame_rate": frame_rate,
        "width": width,
        "height": height
    }
   
    return total_frames, frame_rate, width, height, metadata

if __name__ == "__main__":
    # Parse command line arguments
    from tqdm import tqdm

    parser = argparse.ArgumentParser(description="Extract frames from a video file and save them as images in a folder.")
    parser.add_argument("--video_folder", default = "/dockerx/share/DynamiCrafter/data/AVSync15/test", help="Path to the video file.")
    parser.add_argument("--output_folder", default = "/dockerx/local/DynamiCrafter/data/AVSync15/test_frames", help="Path to the folder where the frames will be saved.")
    args = parser.parse_args()

    # Extract frames from the video file

    training_video = args.video_folder
    output_folder = args.output_folder

    metadata_json = os.path.join(output_folder, "metadata.json")
    all_metadata = {}
    for category in tqdm(os.listdir(training_video)):
        for video in tqdm(os.listdir(os.path.join(training_video, category)), leave=False):
            if not video.endswith(".mp4"):
                continue
            video_path = os.path.join(training_video, category, video)
            output_path = os.path.join(output_folder, category, video.split(".")[0])
            os.makedirs(output_path, exist_ok=True)
            total_frames, frame_rate, width, height, metadata = video_to_frames(video_path, output_path)
            
            all_metadata[video] = metadata
    with open(metadata_json, "w") as f:
        json.dump(all_metadata, f, indent=4)


