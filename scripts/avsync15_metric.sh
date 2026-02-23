# !/bin/sh
# Evaluate generated videos using AVSync15 metrics
#
# Usage: bash scripts/avsync15_metric.sh
#
# Environment variables (optional):
#   AVSYNC15_ROOT       - AVSync15 dataset root (default: ./data/AVSync15)
#   AVSYNC_CKPT         - AVSync checkpoint path (default: /dockerx/groups/KeyVID_hf_model/avsync/.../checkpoint-40000)
#
# Example with custom paths:
#   export AVSYNC15_ROOT=/path/to/AVSync15
#   export AVSYNC_CKPT=/path/to/avsync/checkpoint
#   bash scripts/avsync15_metric.sh

# Get project root
AVSYNC15_ROOT=./data/AVSync15
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

GENERATED_VIDEO_ROOT=/dockerx/groups/KeyVID_publish/outputs/repo/DynamiCrafter/save/asva/asva_12_kf_interp/reproduce_interp_audio_7.5_img_2.0_kf_7.5/ASVA
RESULT_SAVE_PATH=outputs/repo/DynamiCrafter/save/asva/asva_12_kf_interp/reproduce_interp_audio_7.5_img_2.0_kf_7.5/metrics/eval_result.json
# GENERATED_VIDEO_ROOT=$1
# RESULT_SAVE_PATH=$2
DATASET_ROOT="${AVSYNC15_ROOT:-${PROJECT_ROOT}/data/AVSync15}"
AVSYNC_CKPT="${AVSYNC_CKPT:-/dockerx/groups/KeyVID_hf_model/avsync/vggss_sync_contrast_12/ckpts/checkpoint-40000}"
GPU_ID=${3:-0}

if [ -z "$GENERATED_VIDEO_ROOT" ] || [ -z "$RESULT_SAVE_PATH" ]; then
    echo "Usage: bash scripts/avsync15_metric.sh <generated_video_root> <result_save_path> [gpu_id]"
    exit 1
fi

CUDA_VISIBLE_DEVICES=$GPU_ID python -W ignore scripts/evaluation/animation_eval.py \
    --dataset AVSync15 \
    --dataset_root $DATASET_ROOT \
    --generated_video_root $GENERATED_VIDEO_ROOT \
    --result_save_path $RESULT_SAVE_PATH \
    --avsync_ckpt $AVSYNC_CKPT \
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

