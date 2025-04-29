import argparse
import torch
import sys
sys.path.append(".")
from asva.pipeline_audio_cond_animation_uniform_frame import run_inference as run_inference_uniform
from asva.pipeline_audio_cond_animation_keyframe import run_inference as run_inference_keyframe
from asva.pipeline_audio_cond_animation_keyframe_add_idx import run_inference as run_inference_keyframe_add_idx
# from asva.pipeline_audio_cond_animation_freenoise import run_inference as run_inference_freenoise
from asva.pipeline_audio_cond_animation_freenoise_inpainting import run_inference as run_inference_freenoise
from asva.pipeline_audio_cond_animation_48 import run_inference as run_inference_48
from asva.pipeline_audio_cond_animation_48_inpainting import run_inference as run_inference_48_inpainting
from asva.pipeline_audio_cond_animation_60 import run_inference as run_inference_60
from asva.pipeline_audio_cond_animation_no_audio import run_inference as run_inference_no_audio
from asva.pipeline_dynamicrafter import run_inference as run_inference_dynamicrafter


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--exp_root", type=str, default="")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--dataset", type=str, default="AVSync15")
    parser.add_argument("--image_h", type=int, default=256)
    parser.add_argument("--image_w", type=int, default=256)
    parser.add_argument("--video_fps", type=int, default=6)
    parser.add_argument("--video_num_frame", type=int, default=12)
    parser.add_argument("--num_clips_per_video", type=int, default=3)
    parser.add_argument("--audio_guidance_scale", type=float, default=4.0)
    parser.add_argument("--text_guidance_scale", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)
    

    parser.add_argument("--ddim_steps", type=int, default=50, help="steps of ddim if positive, otherwise use DDPM",)
    parser.add_argument("--ddim_eta", type=float, default=1.0, help="eta for ddim sampling (0.0 yields deterministic sampling)",)
    parser.add_argument("--bs", type=int, default=1, help="batch size for inference, should be one")
    parser.add_argument("--height", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--width", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--frame_stride", type=int, default=3, help="frame stride control for 256 model (larger->larger motion), FPS control for 512 or 1024 model (smaller->larger motion)")
    parser.add_argument("--unconditional_guidance_scale", type=float, default=1.0, help="prompt classifier-free guidance")
    parser.add_argument("--seed", type=int, default=123, help="seed for seed_everything")
    parser.add_argument("--video_length", type=int, default=16, help="inference video length")
    parser.add_argument("--negative_prompt", action='store_true', default=False, help="negative prompt")
    parser.add_argument("--text_input", action='store_true', default=False, help="input text to I2V model or not")
    parser.add_argument("--multiple_cond_cfg", action='store_true', default=False, help="use multi-condition cfg or not")
    parser.add_argument("--cfg_img", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--cfg_audio", type=float, default=None, help="guidance scale for image conditioning")
    parser.add_argument("--timestep_spacing", type=str, default="uniform", help="The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.")
    parser.add_argument("--guidance_rescale", type=float, default=0.0, help="guidance rescale in [Common Diffusion Noise Schedules and Sample Steps are Flawed](https://huggingface.co/papers/2305.08891)")
    parser.add_argument("--perframe_ae", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--interp", action='store_true', default=False, help="if we use per-frame AE decoding, set it to True to save GPU memory, especially for the model of 576x1024")
    parser.add_argument("--loop", action='store_true', default=False, help="generate looping videos or not")
    parser.add_argument("--rank", type=int, default=0, help="rank of current gpu")
    parser.add_argument("--swap_audio", action='store_true', default=False, help="swap audio or not")
    parser.add_argument("--inpainting", action='store_true')
    parser.add_argument("--inpainting_end_step", type=int, default=0, help="0 - 90",)

    args = parser.parse_args()
    
    
    print(
        f"########################################\n"
        f"Evaluating Audio-Cond Animation model on\n"
        f"dataset: {args.dataset}\n"
        f"exp: {args.exp_root}\n"
        f"checkpoint: {args.checkpoint}\n"
        f"########################################"
    )
    # generate_videos_for_dataset(
    #     exp_root=args.exp_root,
    #     checkpoint=args.checkpoint,
    #     dataset=args.dataset,
    #     image_size=(args.image_h, args.image_w),
    #     video_fps=args.video_fps,
    #     video_num_frame=args.video_num_frame,
    #     num_clips_per_video=args.num_clips_per_video,
    #     audio_guidance_scale=args.audio_guidance_scale,
    #     text_guidance_scale=args.text_guidance_scale,
    #     random_seed=args.random_seed,
    #     device=torch.device("cuda"),
    #     dtype=torch.float32
    # )

    if 'panda' in args.config:
        print("Running inference without audio")
        run_inference_no_audio(args, gpu_num=8, gpu_no=args.rank)
    elif 'freenoise' in args.config:
        print("Running with freenoise")
        run_inference_freenoise(args, gpu_num=8, gpu_no=args.rank)
    elif args.inpainting:
        print("Running with inpainting")
        # run_inference_48_inpainting(args, gpu_num=7, gpu_no=args.rank)
        run_inference_48_inpainting(args, gpu_num=8, gpu_no=args.rank)
    elif 'uniform' in args.config:
        print("Running with uniform frame")
        run_inference_uniform(args, gpu_num=8, gpu_no=args.rank)
    elif 'add_idx' in args.config:
        print("Running with keyframe add idx")
        run_inference_keyframe_add_idx(args, gpu_num=8, gpu_no=args.rank)
    elif 'keyframe' in args.config:
        print("Running with keyframe")
        run_inference_keyframe(args, gpu_num=8, gpu_no=args.rank)
    elif 'inference_512_asva_12' in args.config:
        print("Running with 12 frames")
        run_inference(args, gpu_num=8, gpu_no=args.rank)
    elif '48' in args.config:
        print("Running with 48 frames")
        run_inference_48(args, gpu_num=8, gpu_no=args.rank)
    elif 'inference_512_v1.0.yaml' in args.config:
        print("Running with dynamicrafter")
        run_inference_dynamicrafter(args, gpu_num=8, gpu_no=args.rank)
    else:
        print("Running inference with audio")
        run_inference_uniform(args, gpu_num=8, gpu_no=args.rank)
    

