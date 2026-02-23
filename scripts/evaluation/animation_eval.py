import argparse
import sys
# sys.path.append('./')
from avgen.evaluations.eval import evaluate_generation_results
from avgen.data.utils import get_evaluation_data


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--dataset", type=str, default="AVSync15", help="Dataset name")
	parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root directory (default: ./data/AVSync15)")
	parser.add_argument("--generated_video_root", type=str, default="")
	parser.add_argument("--num_clips_per_video", type=int, default=3)
	parser.add_argument("--result_save_path", type=str, default="")
	parser.add_argument("--avsync_ckpt", type=str, default="/dockerx/groups/KeyVID_hf_model/avsync/vggss_sync_contrast_12/ckpts/checkpoint-40000")

	parser.add_argument("--image_h", type=int, default=256)
	parser.add_argument("--image_w", type=int, default=256)
	parser.add_argument("--video_fps", type=int, default=6)
	parser.add_argument("--video_num_frame", type=int, default=12)
	parser.add_argument("--keyframe_only", action="store_true")

	
	parser.add_argument("--eval_fid", action="store_true")
	parser.add_argument("--eval_fvd", action="store_true")
	parser.add_argument("--eval_clipsim", action="store_true")
	parser.add_argument("--eval_relsync", action="store_true")
	parser.add_argument("--eval_alignsync", action="store_true")
	parser.add_argument("--eval_fvmd", action="store_true")
	parser.add_argument("--export_video", action="store_true")
	
	
	parser.add_argument("--record_instance_metrics", action="store_true")
	
	args = parser.parse_args()
	
	
	video_root, video_names, categories, video_path_type = get_evaluation_data(args.dataset, args.dataset_root)
	
	
	print(
		f"########################################\n"
		f"# Evaluating videos between\n"
		f"# Groundtruth: {video_root}\n"
		f"# Generated: {args.generated_video_root}\n"
		f"########################################"
	)

	result_dict = evaluate_generation_results(
		groundtruth_video_root=video_root,
		groundtruth_video_names=video_names,
		groundtruth_categories=categories,
		num_clips_per_video=args.num_clips_per_video,
		generated_video_root=args.generated_video_root,
		result_save_path=args.result_save_path,
		avsync_ckpt=args.avsync_ckpt,
		image_size=(args.image_h, args.image_w),
		video_fps=args.video_fps,
		video_num_frame=args.video_num_frame,
		keyframe_only=args.keyframe_only,
		eval_fid=args.eval_fid,
		eval_fvd=args.eval_fvd,
		eval_clipsim=args.eval_clipsim,
		eval_relsync=args.eval_relsync,
		eval_alignsync=args.eval_alignsync,
		eval_fvmd=args.eval_fvmd,
		record_instance_metrics=args.record_instance_metrics,
		export_video=args.export_video,
	)
	print("########################################")
	print("# Resulve saved to ", args.result_save_path)
	print("########################################")
	for metric in [
		'FID', "FVD", "IA_mean", "IA_std", "IT_mean", "IT_std", "RelSync_mean", "RelSync_std", "AlignSync_mean", "AlignSync_std"
	]:
		if metric in result_dict:
			print(f"{metric}:  {result_dict[metric]}")