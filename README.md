# KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation

[![arXiv](https://img.shields.io/badge/arXiv-2504.09656-b31b1b.svg)](https://arxiv.org/pdf/2504.09656) ![License](https://img.shields.io/github/license/XingruiWang/SuperCLEVR-Physics)

Offical code of paper KeyVID: Keyframe-Aware Video Diffusion for Audio-Synchronized Visual Animation.

*Code is still updating.*

# Environment


# Inference

1. Keyframe localization

```bash
cd motion_scores/network
python main.py --mode predict

```

2. Keyframe generator

```bash
bash script/run_ASVA_evaluation.sh asva_12_kf
```

```bash
config='configs/inference_512_asva_12_keyframe_new_add_idx.yaml'
exp_root=${save_root}'/ver_add_idx_add_fps/keyframes'
checkpoint='checkpoints/keyframe_generation/best_checkpoint.ckpt'
```


3. Interpolation

```bash
bash script/run_ASVA_evaluation.sh asva_12_kf_interp
```

```bash
config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
exp_root=${save_root}'/ver_add_idx_add_fps/interpolation/'
checkpoint='checkpoint='checkpoints/interpolation/best_checkpoint.ckpt'
```
