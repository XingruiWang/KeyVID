#!/bin/bash
# Check and display current path configuration

echo "========================================"
echo "KeyVID Path Configuration Check"
echo "========================================"
echo ""

# Get project root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Load environment if .env exists
if [ -f "${SCRIPT_DIR}/.env" ]; then
    echo "✓ Loading environment from scripts/.env"
    source "${SCRIPT_DIR}/.env"
    echo ""
fi

# Check required variable
if [ -z "$CHECKPOINT_ROOT" ]; then
    echo "❌ ERROR: CHECKPOINT_ROOT is not set!"
    echo ""
    echo "Please set CHECKPOINT_ROOT environment variable:"
    echo "  export CHECKPOINT_ROOT=${PROJECT_ROOT}/checkpoint/KeyVID"
    echo ""
    echo "Or create scripts/.env file:"
    echo "  cp scripts/env.example scripts/.env"
    echo "  # Edit scripts/.env to set CHECKPOINT_ROOT"
    echo "  source scripts/.env"
    echo ""
    echo "========================================"
    exit 1
fi

# Set defaults for optional paths
AVSYNC15_ROOT="${AVSYNC15_ROOT:-${PROJECT_ROOT}/data/AVSync15}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/outputs}"

# Checkpoint paths
KF_CHECKPOINT="${CHECKPOINT_ROOT}/keyframe_generation/epoch=859-step=10320.ckpt"
INTERP_CHECKPOINT="${CHECKPOINT_ROOT}/asva_12_kf_interp/epoch=1479-step=17760.ckpt"
PRETRAINED_CHECKPOINT="${CHECKPOINT_ROOT}/dynamicrafter_512_v1/model.ckpt"

# Data paths
AVSYNC15_TRAIN_DIR="${AVSYNC15_ROOT}/train"
AVSYNC15_VIDEOS="${AVSYNC15_ROOT}/videos"
AVSYNC15_TEST_LIST="${AVSYNC15_ROOT}/test.txt"

echo "Project Root:"
echo "  $PROJECT_ROOT"
echo ""

echo "Core Paths:"
echo "  CHECKPOINT_ROOT:    $CHECKPOINT_ROOT"
echo "  AVSYNC15_ROOT:      $AVSYNC15_ROOT"
echo "  DATA_ROOT:          $DATA_ROOT"
echo ""

echo "Checkpoint Files:"
check_file() {
    if [ -f "$1" ]; then
        size=$(du -h "$1" 2>/dev/null | cut -f1)
        echo "  ✓ $1 ($size)"
    else
        echo "  ✗ $1 (NOT FOUND)"
    fi
}

check_file "$KF_CHECKPOINT"
check_file "$INTERP_CHECKPOINT"
check_file "$PRETRAINED_CHECKPOINT"
echo ""

echo "Data Directories:"
check_dir() {
    if [ -d "$1" ]; then
        count=$(find "$1" -maxdepth 1 -type f 2>/dev/null | wc -l)
        echo "  ✓ $1 ($count files)"
    else
        echo "  ✗ $1 (NOT FOUND)"
    fi
}

check_dir "$AVSYNC15_TRAIN_DIR"
check_dir "$AVSYNC15_VIDEOS"
check_file "$AVSYNC15_TEST_LIST"
echo ""

echo "========================================"
echo "Configuration Status:"
missing_count=0

for path in "$KF_CHECKPOINT" "$AVSYNC15_VIDEOS" "$AVSYNC15_TEST_LIST"; do
    if [ ! -e "$path" ]; then
        ((missing_count++))
    fi
done

if [ $missing_count -eq 0 ]; then
    echo "✓ All required paths for keyframe generation are ready!"
    echo ""
    echo "You can now run:"
    echo "  bash scripts/generation.sh asva_12_kf"
else
    echo "✗ $missing_count required path(s) missing."
    echo ""
    echo "Required for keyframe generation:"
    echo "  - $KF_CHECKPOINT"
    echo "  - $AVSYNC15_VIDEOS"
    echo "  - $AVSYNC15_TEST_LIST"
    echo ""
    echo "To customize paths, create scripts/.env file:"
    echo "  cp scripts/env.example scripts/.env"
    echo "  # Edit scripts/.env with your paths"
    echo "  source scripts/.env"
fi
echo "========================================"
