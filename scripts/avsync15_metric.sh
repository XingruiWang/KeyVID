# !/bin/sh
# Usage: bash scripts/evaluation/animation_test_avsync15.sh <generated_video_root> <result_save_path> [gpu_id]
#
# Example:
#   bash scripts/evaluation/animation_test_avsync15.sh \
#     /path/to/generated/ASVA_resave \
#     /path/to/metrics/result.json

GENERATED_VIDEO_ROOT=$1
RESULT_SAVE_PATH=$2
GPU_ID=${3:-0}

if [ -z "$GENERATED_VIDEO_ROOT" ] || [ -z "$RESULT_SAVE_PATH" ]; then
    echo "Usage: bash scripts/evaluation/animation_test_avsync15.sh <generated_video_root> <result_save_path> [gpu_id]"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore scripts/evaluation/animation_eval.py \
    --dataset AVSync15 \
    --generated_video_root $GENERATED_VIDEO_ROOT \
    --result_save_path $RESULT_SAVE_PATH \
    --num_clips_per_video 3 \
    --image_h 256 \
    --image_w 256 \
    --video_num_frame 12 \
    --video_fps 6 \
    --eval_fid \
    --eval_fvd \
    --eval_clipsim \
    --eval_relsync \
    --eval_alignsync \
    --record_instance_metrics

