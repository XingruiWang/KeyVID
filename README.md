<h1 align="center">🎬 KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation</h1>

<p align="center">
  <a href="https://xingruiwang.github.io/projects/KeyVID/">
    <img src="https://img.shields.io/badge/Project%20Page-KeyVID-0a7aca?logo=globe&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/pdf/2504.09656">
    <img src="https://img.shields.io/badge/arXiv-2504.09656-b31b1b.svg" alt="arXiv">
  </a>
  <a href="https://huggingface.co/RyanWW/KeyVID">
    <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-blue" alt="Model">
  </a>
  <a href="https://huggingface.co/datasets/RyanWW/KeyVID_data">
    <img src="https://img.shields.io/badge/Hugging%20Face-Data-orange" alt="Data">
  </a>
  <img src="https://img.shields.io/github/license/XingruiWang/KeyVID" alt="License">
</p>

<p align="center">
  <a href="https://xingruiwang.github.io/projects/KeyVID/">
    <img src="https://xingruiwang.github.io/projects/KeyVID/static/videos/teaser_small.gif" width="80%" alt="KeyVID Teaser">
  </a>
</p>

Official repository for **KeyVID** - a unified diffusion framework that generates temporally coherent videos conditioned on audio, guided by adaptive keyframe localization.

---

## 📦 Release Plan

- [ ] Keyframe Localization Code — Coming soon  
- [x] Keyframe Generation Code — Released ✅
- [x] Interpolation Code — Released ✅
- [ ] Training Code — Coming soon
- [ ] Pre-trained Checkpoints — Coming soon (HuggingFace)

---

## 🚀 Quick Start

```bash
# 1. Clone and install
git clone https://github.com/XingruiWang/KeyVID.git
cd KeyVID
pip install -r requirements.txt

# 2. Create directories and download checkpoints
mkdir -p checkpoint/KeyVID/keyframe_generation checkpoint/KeyVID/asva_12_kf_interp
mkdir -p data/AVSync15/videos

# Download checkpoints (coming soon on HuggingFace)
# Place in: checkpoint/KeyVID/keyframe_generation/epoch=859-step=10320.ckpt
#           checkpoint/KeyVID/asva_12_kf_interp/epoch=1479-step=17760.ckpt

# 3. Prepare test data
# Place videos in: data/AVSync15/videos/
# Create: data/AVSync15/test.txt (one video name per line, no extension)

# 4. Set checkpoint path (REQUIRED)
export CHECKPOINT_ROOT=./checkpoint/KeyVID

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
- Evaluation requires AVSync checkpoint: `/dockerx/groups/KeyVID_hf_model/avsync/vggss_sync_contrast_12/ckpts/checkpoint-40000`

---

## 📂 Directory Structure

```
KeyVID/                                     # Project root
├── checkpoint/                             # Model checkpoints (git-ignored)
│   └── KeyVID/                             # Checkpoint root (set via CHECKPOINT_ROOT)
│       ├── keyframe_generation/
│       │   └── epoch=859-step=10320.ckpt  # ⚠️ REQUIRED
│       └── asva_12_kf_interp/
│           └── epoch=1479-step=17760.ckpt # ⚠️ REQUIRED
│
├── .checkpoints/                           # Downloaded by Imagebind
│
├── data/                                   # Input datasets (git-ignored)
│   └── AVSync15/                           # Dataset
│       ├── videos/                         # ⚠️ REQUIRED
│       └── test.txt                        # ⚠️ REQUIRED
│
├── outputs/                                # Inference outputs (git-ignored)
├── save_results/                           # Saved evaluation results (git-ignored)
│
├── scripts/                                # Inference & evaluation scripts
│   ├── generation.sh                       # Main inference script
│   ├── avsync15_metric.sh                  # Evaluation script
│   └── check_paths.sh                      # Setup verification
├── configs/                                # Configuration files
├── imagebind/                              # ImageBind model code
├── lvdm/                                   # Model code
├── utils/                                  # Utility modules
├── motion_scores/                          # Motion scoring tools
├── main/                                   # Main entry points
├── Dockerfile                              # Docker build config
├── docker-compose.yml                      # Docker Compose config
├── requirements.txt                        # Python dependencies
└── README.md
```

**External Checkpoints** (for evaluation only):
- AVSync checkpoint: `/dockerx/groups/KeyVID_hf_model/avsync/vggss_sync_contrast_12/ckpts/checkpoint-40000`
  - Used by `scripts/avsync15_metric.sh` for audio-visual synchronization metrics
  - Can be customized via `AVSYNC_CKPT` environment variable

---

## ⚙️ Installation

### Option 1: Conda/Pip

```bash
# Clone repository
git clone https://github.com/XingruiWang/KeyVID.git
cd KeyVID

# Create environment
conda create -n keyvid python=3.10
conda activate keyvid

# Install PyTorch (CUDA example)
pip install torch torchvision torchaudio

# Or for AMD ROCm:
pip install torch==2.3.0+rocm6.0 torchaudio==2.3.0+rocm6.0 \
    --index-url https://download.pytorch.org/whl/rocm6.0

# Install dependencies
pip install -r requirements.txt

# Install envsubst (for config processing)
# Ubuntu/Debian: sudo apt-get install gettext
# macOS: brew install gettext
```

### Option 2: Docker (Recommended for ROCm)

```bash
# Build and run
docker-compose up -d
docker-compose exec keyvid bash

# Inside container
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

## 🎬 Inference

KeyVID uses a two-stage pipeline:

### Stage 1: Generate Keyframes

```bash
bash scripts/generation.sh asva_12_kf
```

- Generates 12-frame videos at 6 FPS
- Results: `outputs/repo/DynamiCrafter/save/asva/asva_12_kf_add_idx_add_fps/`

### Stage 2: Interpolate

```bash
bash scripts/generation.sh asva_12_kf_interp
```

- Generates 48-frame videos at 24 FPS
- Uses keyframes from Stage 1
- Results: `outputs/repo/DynamiCrafter/save/asva/asva_12_kf_interp/`

**Note**: Run Stage 1 before Stage 2.

---

## 🔧 Configuration

### Required: Set Checkpoint Path

You **must** set `CHECKPOINT_ROOT` before running:

```bash
# Option 1: Direct export
export CHECKPOINT_ROOT=./checkpoint/KeyVID
bash scripts/generation.sh asva_12_kf

# Option 2: Use config file (recommended)
cp scripts/env.example scripts/.env
# Edit scripts/.env to set CHECKPOINT_ROOT
source scripts/.env
bash scripts/generation.sh asva_12_kf
```

### Optional: Custom Data Paths

By default, scripts use:
- Data: `./data/AVSync15/`
- Outputs: `./outputs/`

To customize:
```bash
export CHECKPOINT_ROOT=./checkpoint/KeyVID
export AVSYNC15_ROOT=/path/to/data
export DATA_ROOT=/path/to/outputs
bash scripts/generation.sh asva_12_kf
```

### Adjust GPU Count

Edit `scripts/generation.sh` line 113:
```bash
for ((i=0; i<N; i++)); do  # Change N to your GPU count
```

---

## 📥 Download Checkpoints

**Status**: Coming soon on HuggingFace.

### Inference Checkpoints

Required files (place under `checkpoint/KeyVID/`):
- `checkpoint/KeyVID/keyframe_generation/epoch=859-step=10320.ckpt` (~2.5GB)
- `checkpoint/KeyVID/asva_12_kf_interp/epoch=1479-step=17760.ckpt` (~2.5GB)

Then set the checkpoint root:
```bash
export CHECKPOINT_ROOT=./checkpoint/KeyVID
```

### Evaluation Checkpoint (Optional)

For quantitative evaluation metrics (AlignSync, RelSync), you need:
- **Location**: `/dockerx/groups/KeyVID_hf_model/avsync/vggss_sync_contrast_12/ckpts/checkpoint-40000`
- **Used by**: `scripts/avsync15_metric.sh`
- **Configure via**: `export AVSYNC_CKPT=/path/to/avsync/checkpoint-40000`

---

## 🐛 Troubleshooting

### Checkpoint not found
```bash
bash scripts/check_paths.sh  # Verify paths
```

### Out of memory
Reduce GPU count in `scripts/generation.sh` (line 113)

### envsubst not found
```bash
# Ubuntu/Debian
sudo apt-get install gettext

# macOS
brew install gettext
```

### ROCm not detected (Docker)
```bash
# Inside container
rocm-smi
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📊 Evaluation

Evaluate generated videos with quantitative metrics:

```bash
# Set dataset path (optional if using default ./data/AVSync15)
export AVSYNC15_ROOT=./data/AVSync15

# Run evaluation
bash scripts/avsync15_metric.sh
```

**Metrics computed**:
- FID (Fréchet Inception Distance)
- FVD (Fréchet Video Distance)
- CLIP-Sim (Image-Audio and Image-Text similarity)
- RelSync (Relative audio-visual synchronization)
- AlignSync (Alignment synchronization)

**Requirements**:
- AVSync checkpoint: `/dockerx/groups/KeyVID_hf_model/avsync/vggss_sync_contrast_12/ckpts/checkpoint-40000`
- Ground truth videos: `data/AVSync15/videos/`
- Generated videos: `outputs/repo/DynamiCrafter/save/asva/.../samples/`

**Custom paths**:
```bash
export AVSYNC15_ROOT=/path/to/AVSync15
export AVSYNC_CKPT=/path/to/avsync/checkpoint-40000
bash scripts/avsync15_metric.sh
```

---

## 🎓 Training

**Status**: Training code coming soon.

Two-stage training:
1. **Keyframe Generation**: Audio + Image + Frame indices → Keyframes
2. **Interpolation**: Keyframes + Audio → Full video

Configurations available in `configs/training/`.

---

## 📚 Citation

```bibtex
@article{wang2025keyvid,
  title={KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation},
  author={Wang, Xingrui and Liu, Jiang and Wang, Ze and Yu, Xiaodong and Wu, Jialian and Sun, Ximeng and Su, Yusheng and Yuille, Alan and Liu, Zicheng and Barsoum, Emad},
  journal={arXiv preprint arXiv:2504.09656},
  year={2025}
}
```

---

## 📄 License

See [LICENSE](LICENSE) file for details.
