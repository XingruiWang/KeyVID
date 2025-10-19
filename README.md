<h1 align="center">🎬 KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation</h1>

<p align="center">
  <a href="https://xingruiwang.github.io/projects/KeyVID/">
    <img src="https://img.shields.io/badge/Project%20Page-KeyVID-0a7aca?logo=globe&logoColor=white" alt="Project Page Badge">
  </a>
  <a href="https://arxiv.org/pdf/2504.09656">
    <img src="https://img.shields.io/badge/arXiv-2504.09656-b31b1b.svg" alt="arXiv Badge">
  </a>
  <img src="https://img.shields.io/github/license/XingruiWang/KeyVID" alt="License Badge">
</p>



<p align="center">
  <a href="https://xingruiwang.github.io/projects/KeyVID/static/videos/teaser.mp4">
    <img src="https://xingruiwang.github.io/projects/KeyVID/static/videos/teaser_small.gif" width="80%" alt="KeyVID Teaser">
  </a>
</p>

<p align="center">
<video id="teaser-video" autoplay preload muted loop playsinline>
  <source src="./static/videos/teaser.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>
</p>
Official repository for **KeyVID**, presented in  **“KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation.”**  
This work introduces a unified diffusion framework that generates temporally coherent videos conditioned on audio, guided by adaptive keyframe localization.

---

## 📦 Release Plan

- [ ] **Keyframe Localization** — Coming soon  
- [x] **Keyframe Generation** — Released  
- [x] **Interpolation Model** — Released 
- [ ] **Training Code** — Coming soon
- [ ] Checkpoints

---

## ⚙️ Environment Setup

We recommend using **Python 3.10+** and **PyTorch ≥ 2.1**.

```bash
# Clone the repository
git clone https://github.com/XingruiWang/KeyVID.git
cd KeyVID

# Create environment
conda create -n keyvid python=3.10
conda activate keyvid

# Install dependencies
pip install -r requirements.txt
```


## 🚀 Inference

### 1️⃣ Keyframe Localization
Detect audio-synchronized keyframes:

```bash
bash scripts/run_ASVA_evaluation.sh asva_12_kf
```

---

### 2️⃣ Keyframe Generation
Generate keyframes aligned with localized timestamps:

```bash
bash scripts/run_ASVA_evaluation.sh asva_12_kf
```

**Configuration example:**
```bash
config="configs/inference_512_asva_12_keyframe_new_add_idx.yaml"
exp_root="${save_root}/ver_add_idx_add_fps/keyframes"
checkpoint="checkpoints/keyframe_generation/best_checkpoint.ckpt"
```

---

### 3️⃣ Interpolation
Generate smooth video transitions between keyframes:

```bash
bash scripts/run_ASVA_evaluation.sh asva_12_kf_interp
```

**Configuration example:**
```bash
config="configs/inference_512_asva_12_keyframe_kf_freenoise.yaml"
exp_root="${save_root}/ver_add_idx_add_fps/interpolation/"
checkpoint="checkpoints/interpolation/best_checkpoint.ckpt"
```

---

## 📈 Evaluation

Quantitative evaluation (e.g., **FID**, **FVD**, **AlignSync**, **RelSync**) scripts will be added soon.  
You can also visualize the output videos in the `outputs/` directory for qualitative comparison.

---

## 📚 Citation

If you find this project useful, please cite:

```bibtex
@article{wang2025keyvid,
  title={KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation},
  author={Wang, Xingrui and Liu, Jiang and Wang, Ze and Yu, Xiaodong and Wu, Jialian and Sun, Ximeng and Su, Yusheng and Yuille, Alan and Liu, Zicheng and Barsoum, Emad},
  journal={arXiv preprint arXiv:2504.09656},
  year={2025}
}
```


