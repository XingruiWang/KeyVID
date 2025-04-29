import cv2
import imageio

path = "/dockerx/local/DynamiCrafter/data/AVSync15/test_frames/hammering/lptpDgCE0N4_000083_000093_Scene-002-1"

# Read the frames
frames = []
for i in range(0, 48, 1):
    frame_path = f"{path}/frame_{i:04d}.jpg"
    frame = cv2.imread(frame_path)
    frames.append(frame[:, :, ::-1])

# Save frames as GIF
output_path = "fps_ori.gif"
imageio.mimsave(output_path, frames, fps=24)
