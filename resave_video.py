#!/usr/bin/env python3
import cv2
import subprocess
from pathlib import Path
from tqdm import tqdm

IN_ROOT  = Path("/dockerx/groups/asva_12_kf-interp/reproduce_interp_audio_9.0_img_2.0_kf_9.0/ASVA")
OUT_ROOT = IN_ROOT.with_name(IN_ROOT.name + "_resave")  # -> ASVA_resave

def run_ffmpeg_mux(video_noaudio: Path, original_with_audio: Path, out_final: Path) -> bool:
    """
    Mux original audio stream from `original_with_audio` with the processed video stream
    from `video_noaudio`. Copies streams without re-encoding.
    Works even if there's no audio (the `?` makes it optional).
    """
    out_final.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_noaudio),          # 0: processed video (no/any audio)
        "-i", str(original_with_audio),    # 1: original (has audio)
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "copy",                    # no re-encode for video
        "-c:a", "copy",                    # no re-encode for audio
        str(out_final)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return True
    except subprocess.CalledProcessError:
        print(f"[ffmpeg mux failed] {out_final}")
        return False

def fix_tail_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[skip] cannot open: {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # widely compatible

    # Read all frames
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < 4:
        print(f"[skip] <4 frames: {video_path}")
        return

    # Replace last 3 frames with the 4th-from-last (index -4)
    # frames[-2:] = [frames[-3]] * 2
    frames[-1] = frames[-2]

    # Build output paths robustly (mirror folder structure under OUT_ROOT)
    rel = video_path.relative_to(IN_ROOT)
    out_dir = (OUT_ROOT / rel.parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp_noaudio = out_dir / (rel.stem + ".noaudio.mp4")
    out_final   = out_dir / (rel.stem + ".mp4")

    # Write processed video (no audio) with OpenCV
    vw = cv2.VideoWriter(str(tmp_noaudio), fourcc, fps, (w, h))
    for f in frames:
        vw.write(f)
    vw.release()

    # Mux original audio back in
    if run_ffmpeg_mux(tmp_noaudio, video_path, out_final):
        tmp_noaudio.unlink(missing_ok=True)
        print(f"[done] {out_final}")
    else:
        print(f"[kept no-audio version] {tmp_noaudio}")

def batch_fix(in_root: Path = IN_ROOT):
    videos = sorted(in_root.rglob("*.mp4"))
    print(f"Found {len(videos)} videos under {in_root}")
    for vp in tqdm(videos):
        fix_tail_frames(vp)

if __name__ == "__main__":
    batch_fix()
