<h1 align="center">KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation</h1>

<p align="center">
  <em>Anonymous submission — code for review purposes only</em>
</p>

---

## Quick Start

```bash
# 1. Install
cd KeyVID
pip install -r requirements.txt

# 2. Create directories and download checkpoints
mkdir -p checkpoint/KeyVID/keyframe_generation checkpoint/KeyVID/interpolation
mkdir -p data/AVSync15/videos

# Download checkpoints (will be released upon acceptance)
# Place in: checkpoint/KeyVID/keyframe_generation/generator_checkpoint.ckpt
#           checkpoint/KeyVID/interpolation/epoch=3669-step=58720.ckpt

# 3. Prepare test data
# Place videos in: data/AVSync15/videos/
# Create: data/AVSync15/test.txt (one video name per line, no extension)

# 4. Set checkpoint path (REQUIRED)
export CHECKPOINT_ROOT=./checkpoint

# 5. Verify setup
bash scripts/check_paths.sh

# 6. Run inference
bash scripts/generation.sh asva_12_kf          # Generate keyframes
bash scripts/generation.sh asva_12_kf_interp   # Interpolate to full video

# 7. (Optional) Run evaluation
bash scripts/avsync15_metric.sh                # Compute metrics
```

**Notes**: 
- If you get "CHECKPOINT_ROOT is not set" error, make sure to export it in step 4
- Evaluation requires AVSync checkpoint (see Evaluation section below)

---

## Directory Structure

```
KeyVID/
├── configs/
│   ├── inference/
│   │   ├── keyframe_generation.yaml
│   │   └── keyframe_interpolation.yaml
│   └── training/
│       ├── keyframe_generation.yaml
│       └── keyframe_interpolation.yaml
│
├── scripts/
│   ├── generation.sh                  # Main inference script
│   ├── train.sh                       # Training script
│   ├── avsync15_metric.sh             # Evaluation script
│   ├── check_paths.sh                 # Setup verification
│   ├── env.example                    # Example env config
│   └── evaluation/
│       ├── animation_gen.py           # Inference entry point
│       ├── animation_eval.py          # Evaluation entry point
│       ├── asva/                      # Keyframe & interpolation pipelines
│       ├── avgen/                     # Generation evaluation modules
│       └── avsync/                    # Audio-visual sync evaluation
│
├── lvdm/                              # Core model code
│   ├── data/                          # Dataset loaders
│   ├── models/                        # Diffusion models & samplers
│   └── modules/                       # UNet, attention, encoders
│
├── imagebind/                         # ImageBind audio encoder
├── motion_scores/                     # Motion scoring (RAFT-based)
│   └── network/                       # Audio-motion prediction network
├── main/                              # Training entry points
│   ├── trainer.py
│   ├── callbacks.py
│   └── utils_*.py
├── utils/                             # Utility modules
│
├── checkpoint/                        # Model checkpoints (git-ignored)
│   └── KeyVID/
│       ├── keyframe_generation/
│       │   └── generator_checkpoint.ckpt
│       └── interpolation/
│           └── interpolation_checkpoint.ckpt
│
├── data/                              # Datasets (git-ignored)
│   └── AVSync15/
│       ├── videos/
│       └── test.txt
├── outputs/                           # Inference outputs (git-ignored)
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### Option 1: Conda/Pip

```bash
cd KeyVID

conda create -n keyvid python=3.10
conda activate keyvid

# Install PyTorch (CUDA)
pip install torch torchvision torchaudio

# Or for AMD ROCm:
pip install torch==2.3.0+rocm6.0 torchaudio==2.3.0+rocm6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.0

pip install -r requirements.txt

# Install envsubst (for config processing)
# Ubuntu/Debian: sudo apt-get install gettext
# macOS: brew install gettext
```

### Option 2: Docker (Recommended for ROCm)

```bash
docker-compose up -d
docker-compose exec keyvid bash

cd /workspace
bash scripts/generation.sh asva_12_kf
```

**Docker GPU Configuration**:
- Update `HSA_OVERRIDE_GFX_VERSION` in `docker-compose.yml` for your GPU
- Check GPU: `rocminfo | grep gfx`

**System Requirements**:
- GPU: 16GB+ VRAM recommended
- Python 3.10+, PyTorch 2.1+
- FFmpeg, envsubst (gettext)

---

## Inference

KeyVID uses a two-stage pipeline:

### Stage 1: Generate Keyframes

```bash
bash scripts/generation.sh asva_12_kf
```

- Generates 12-frame videos at 6 FPS
- Results: `outputs/asva/asva_12_kf_add_idx_add_fps/`

### Stage 2: Interpolate

```bash
bash scripts/generation.sh asva_12_kf_interp
```

- Generates 48-frame videos at 24 FPS
- Uses keyframes from Stage 1

**Note**: Run Stage 1 before Stage 2.

---

## Configuration

### Required: Set Checkpoint Path

You **must** set `CHECKPOINT_ROOT` before running:

```bash
export CHECKPOINT_ROOT=./checkpoint
bash scripts/generation.sh asva_12_kf
```

Or use a config file:
```bash
cp scripts/env.example scripts/.env
# Edit scripts/.env to set CHECKPOINT_ROOT
source scripts/.env
bash scripts/generation.sh asva_12_kf
```

### Optional: Custom Data Paths

```bash
export CHECKPOINT_ROOT=./checkpoint
export AVSYNC15_ROOT=/path/to/AVSync15
export DATA_ROOT=/path/to/outputs
bash scripts/generation.sh asva_12_kf
```

### Adjust GPU Count

Edit `scripts/generation.sh` line 150:
```bash
for ((i=0; i<8; i++)); do  # Change 8 to your GPU count
```

---

## Download Checkpoints

**Status**: Will be released upon acceptance.

### Inference Checkpoints

Required files (place under `checkpoint/KeyVID/`):
- `checkpoint/KeyVID/keyframe_generation/generator_checkpoint.ckpt`
- `checkpoint/KeyVID/interpolation/epoch=3669-step=58720.ckpt`

A pretrained base model is also needed:
- `checkpoint/dynamicrafter_512_v1/model.ckpt`

### Evaluation Checkpoint (Optional)

For quantitative evaluation metrics (AlignSync, RelSync):
```bash
export AVSYNC_CKPT=/path/to/avsync/checkpoint-40000
```

---

## Training

Two-stage training:

### Stage 1: Keyframe Generation
```bash
export CHECKPOINT_ROOT=./checkpoint
bash scripts/train.sh keyframe_generation
```

### Stage 2: Keyframe Interpolation
```bash
bash scripts/train.sh keyframe_interpolation
```

Training configs are in `configs/training/`. Both stages use 8 GPUs by default (configurable in `scripts/train.sh`).

---

## Evaluation

```bash
export AVSYNC15_ROOT=./data/AVSync15
export AVSYNC_CKPT=/path/to/avsync/checkpoint-40000
bash scripts/avsync15_metric.sh
```

**Metrics computed**:
- FID (Frechet Inception Distance)
- FVD (Frechet Video Distance)
- CLIP-Sim (Image-Audio and Image-Text similarity)
- RelSync (Relative audio-visual synchronization)
- AlignSync (Alignment synchronization)

---

## Troubleshooting

### Checkpoint not found
```bash
bash scripts/check_paths.sh
```

### Out of memory
Reduce GPU count in `scripts/generation.sh` (line 150)

### envsubst not found
```bash
sudo apt-get install gettext   # Ubuntu/Debian
brew install gettext            # macOS
```

---

## License

See [LICENSE](LICENSE) file for details.
