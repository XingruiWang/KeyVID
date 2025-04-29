# DEVICE_ID=$1
DATASET=$1
# exp_root=$1
# iter=$2
# audio_cfg=$3

# CUDA_VISIBLE_DEVICES=4,5,6,7 python -W ignore scripts/asva_animation/animation_gen.py \
# --exp_root ${exp_root} \
# --checkpoint ${iter} \
# --dataset AVSync15 \
# --image_h 256 \
# --image_w 256 \
# --video_fps 6 \
# --video_num_frame 12 \
# --num_clips_per_video 3 \
# --audio_guidance_scale ${audio_cfg} \
# --text_guidance_scale 1.0 \
# --random_seed 0

FS=6


# --config configs/inference_512_avsyn.yaml \
# --checkpoint main/save/training_512_avsyn_qformer_finetune/checkpoints/epoch=619-step=14880.ckpt \
# --exp_root save/training_512_avsyn_qformer_finetune \

# 

# CUDA_VISIBLE_DEVICES=${DEVICE_ID} python -W ignore scripts/evaluation/animation_gen.py \
# --config configs/inference_512_avsyn_12.yaml \
# --exp_root save/training_512_avsyn_qformer_12 \
# --checkpoint /dockerx/local/repo/DynamiCrafter/main/save/training_512_avsyn_qformer_12/checkpoints/epoch=19-step=480.ckpt \
# --dataset AVSync15 \
# --image_h 512 \
# --image_w 512 \
# --video_fps 6 \
# --video_length 12 \
# --num_clips_per_video 3 \
# --audio_guidance_scale 4.0 \
# --text_guidance_scale 1.0 \
# --random_seed 0 \
# --frame_stride ${FS} \
# --unconditional_guidance_scale 7.5 \
# --ddim_steps 50 \
# --ddim_eta 1.0 \
# --text_input \
# --rank ${DEVICE_ID} \
# --timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae

FS=6
video_length=12

# qformer 24 checkpoint /dockerx/share/DynamiCrafter/main/save/training_512_avsyn_qformer_24/checkpoints/epoch=309-step=3720.ckpt \
# no qformer /dockerx/share/DynamiCrafter/main/save/training_512_avsyn_n/checkpoints/epoch=319-step=7680.ckpt
# FS=12
# --config inference_512_avsyn_12_keyframe_with_position_finetune.yaml \
# training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_frameidx_position_embedding
save_root='/dockerx/local/repo/DynamiCrafter/save'

if [ $DATASET == "panda" ]; then
    config='configs/inference_512_panda_12_keyframe.yaml'
    exp_root=${save_root}'/panda/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_frameidx_position_embedding_swin/epoch=6-step=28200.ckpt'
    checkpoint='checkpoints/panda/epoch=6-step=28200.ckpt'

    
elif [ $DATASET == "vggsound" ]; then
    config='configs/inference_512_vgg_sound_12_keyframe.yaml'
    exp_root=${save_root}'/vgg_sound/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/epoch=2-step=13650.ckpt'
    checkpoint='checkpoints/vgg_sound/epoch=2-step=13650.ckpt'

elif [ $DATASET == "asva" ]; then

    # config='configs/inference_512_asva_12_keyframe.yaml'
    # exp_root=${save_root}'/asva/asva_12_split_audio/epoch=689-step=16560'
    # checkpoint='checkpoints/asva_12_split_audio/epoch=689-step=16560.ckpt'
    
    # config='configs/inference_512_avsyn_12_landscapes.yaml'
    # exp_root=${save_root}'/landscapes/epoch=909-step=25480'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_12_uniform_landscapes/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=909-step=25480.ckpt'

    # config='configs/inference_512_avsyn_12_landscapes.yaml'
    # exp_root=${save_root}'/landscapes/epoch=609-step=17080'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_12_uniform_landscapes/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=609-step=17080.ckpt'

    # config='configs/inference_512_avsyn_12_thegreatesthit.yaml'
    # exp_root=${save_root}'/thegreatehits/epoch=1209-step=27830'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/thegreatehits/epoch=1209-step=27830.ckpt'

    # # not trained dynamicrafter
    # config='configs/inference_512_avsyn_12_landscapes.yaml'
    # exp_root=${save_root}'/landscapes/dynamicrafter'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_dynamicrafter_landscapes/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=0-step=22.ckpt'

    # not trained dynamicrafter
    # config='configs/inference_512_avsyn_12_thegreatesthit.yaml'
    # exp_root=${save_root}'/thegreatehits/dynamicrafter'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_dynamicrafter_landscapes/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=0-step=22.ckpt'


    FS=6
    video_length=12

    
elif [ $DATASET == "dynamicrafter" ]; then
    config='configs/inference_512_v1.0.yaml'
    exp_root=${save_root}'/asva/asva_dynamicrafter'
    checkpoint='checkpoints/dynamicrafter_512_v1/model.ckpt'
    FS=6
    video_length=12
elif [ $DATASET == "asva_qformer" ]; then
    config='configs/inference_512_asva_12_keyframe_qformer.yaml'
    exp_root=${save_root}'/asva/asva_12/epoch=229-step=5520'
    checkpoint='checkpoints/asva_12/epoch=229-step=5520.ckpt'



elif [ $DATASET == "asva_48" ]; then
    config='configs/inference_512_asva_48_keyframe.yaml'
    exp_root=${save_root}'/asva/asva_48/epoch=579-step=13920'
    # checkpoint='checkpoints/asva_48_split_audio/epoch=179-step=4320.ckpt'
    checkpoint='checkpoints/asva_48_split_audio/epoch=579-step=13920.ckpt'
    FS=24
    video_length=48


elif [ $DATASET == "asva_48_inpainting" ]; then
    config='configs/inference_512_asva_48_keyframe.yaml'
    exp_root=${save_root}'/asva/asva_48/epoch=579-step=13920-inpainting'
    # exp_root=${save_root}'/asva/asva_48/epoch=579-step=13920-inpainting-uniform'
    # checkpoint='checkpoints/asva_48_split_audio/epoch=179-step=4320.ckpt'
    checkpoint='checkpoints/asva_48_split_audio/epoch=579-step=13920.ckpt'
    FS=24
    video_length=48

elif [ $DATASET == "asva_12_kf" ]; then
    # config='configs/inference_512_asva_12_keyframe_new.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe/epoch=479-step=11520-3'
    # checkpoint='checkpoints/asva_12_kf_split_audio/epoch=479-step=11520.ckpt'

    # config='configs/inference_512_asva_12_keyframe_new_add_idx.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_add_idx/epoch=309-step=3720-uniform'
    # checkpoint='checkpoints/asva_12_kf_add_idx/epoch=309-step=3720.ckpt'

    # config='configs/inference_512_asva_12_keyframe_new_add_idx.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_no_idx/epoch=419-step=5040-2'
    # checkpoint='checkpoints/asva_12_kf_no_idx/epoch=419-step=5040.ckpt'

    # config='configs/inference_512_asva_12_keyframe_new_add_idx.yaml'
    # exp_root=${save_root}'/asva/asva_12_kf_no_idx-countinous-12/epoch=519-step=6240'
    # checkpoint='checkpoints/asva_12_kf_no_idx-countinous-12/epoch=519-step=6240.ckpt'

    config='configs/inference_512_asva_12_keyframe_new_add_idx_real.yaml'
    # config='configs/inference_512_asva_12_keyframe_new_add_idx.yaml'
    # exp_root=${save_root}'/asva/asva_12_kf_no_idx/epoch=849-step=10200-kf'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/asva_12_kf_no_idx-countinous-12/epoch=849-step=10200.ckpt'
    exp_root=${save_root}'/asva/asva_12_kf_add_idx_add_fps/epoch=1319-step=15840-kf'
    # checkpoint='main/save/asva_12_kf_split_audio_add_frameidx-kf_0_2/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=1319-step=15840.ckpt'
    checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_12_kf_split_audio_add_frameidx-kf_0_4-adding-fps/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=859-step=10320.ckpt'


    # config='share/Dynamicrafter_audio/configs/inference_512_avsyn_12.yaml'
    # exp_root=${save_root}'/asva/asva_12_uniform/epoch=849-step=10200'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/asva_12_kf_no_idx-countinous-12/epoch=849-step=10200.ckpt'


    # config='configs/inference_512_asva_12_uniform_no_idx.yaml'
    # exp_root=${save_root}'/asva/asva_12_uniform/epoch=849-step=10200'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/asva_12_kf_no_idx-countinous-12/epoch=849-step=10200.ckpt'



    # config='configs/inference_512_asva_12_keyframe_new_add_idx.yaml'
    # exp_root=${save_root}'/asva/asva_12_uniform/epoch=549-step=6600-no_audio'
    # # checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/asva_12_kf_no_idx-countinous-12/epoch=849-step=10200.ckpt'

    # checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_12_kf_split_audio_no_frameidx-countinous-12-add_kf/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/train_steps/epoch=549-step=6600.ckpt'

    # config='configs/inference_512_asva_12_uniform_no_idx.yaml'
    # exp_root=${save_root}'/asva/asva_dynamicrafter'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_dynamicrafter/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=0-step=1.ckpt'
    

    config='configs/inference_512_asva_12_keyframe_new_add_idx_real.yaml'
    exp_root=${save_root}'/asva/asva_12_kf_add_idx_add_fps/open_domain-kf'
    # checkpoint='main/save/asva_12_kf_split_audio_add_frameidx-kf_0_2/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=1319-step=15840.ckpt'
    checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_12_kf_split_audio_add_frameidx-kf_0_4-adding-fps/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=859-step=10320.ckpt'


    FS=6
    video_length=12
    
elif [ $DATASET == "asva_12_freenoise" ]; then
    config='configs/inference_512_asva_12_keyframe_freenoise.yaml'
    exp_root=${save_root}'/asva/asva_12_split_audio/epoch=689-step=16560_freenoise-2-no-audio'
    checkpoint='checkpoints/asva_12_split_audio/epoch=689-step=16560.ckpt'
    FS=24
    video_length=48

elif [ $DATASET == "asva_12_kf_freenoise" ]; then
    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_split_audio/epoch=479-step=11520-kf_freenoise'
    # checkpoint='checkpoints/asva_12_kf_split_audio/epoch=479-step=11520.ckpt'


    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_add_idx/epoch=189-step=2280-kf_freenoise'
    # checkpoint='checkpoints/asva_12_kf_add_idx/epoch=189-step=2280.ckpt'

    # current best
    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_add_idx/epoch=579-step=6960-kf_freenoise'
    # checkpoint='checkpoints/asva_12_kf_add_idx/epoch=579-step=6960.ckpt'

    

    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_no_frameidx-countinous-12/epoch=519-step=6240-kf_freenoise'
    # checkpoint='checkpoints/asva_12_kf_no_idx-countinous-12/epoch=519-step=6240.ckpt'


    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_add_idx_2/epoch=1029-step=24720-kf_freenoise'
    # checkpoint='checkpoints/asva_12_kf_add_idx_2/epoch=1029-step=24720.ckpt'

    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_add_idx_2/epoch=1199-step=14400-kf_freenoise'
    # checkpoint='checkpoints/asva_12_kf_add_idx_2/epoch=1199-step=14400.ckpt'


    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_add_idx_2/epoch=1199-step=14400-kf_freenoise-inpainting-2'
    # checkpoint='checkpoints/asva_12_kf_add_idx_2/epoch=1199-step=14400.ckpt'

    
    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_freenoise'
    # # checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/asva_12_kf_add_idx_add_fps/epoch=889-step=10680.ckpt'
    # checkpoint='main/save/asva_12_kf_split_audio_add_frameidx-kf_0_4-adding-fps/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=1339-step=16080.ckpt'


    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_kf_no_idx/epoch=849-step=10200-kf_freenoise'
    # checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/asva_12_kf_no_idx-countinous-12/epoch=849-step=10200.ckpt'


    config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    exp_root=${save_root}'/asva/asva_12_kf_no_idx/epoch=849-step=10200-kf_freenoise-jump'
    checkpoint='/dockerx/share/Dynamicrafter_audio/checkpoints/asva_12_kf_no_idx-countinous-12/epoch=849-step=10200.ckpt'

    
    # config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_no_idx/epoch=419-step=5040-kf_freenoise-2'
    # checkpoint='checkpoints/asva_12_kf_no_idx/epoch=419-step=5040.ckpt'


    FS=24
    video_length=48

elif [ $DATASET == "asva_12_kf_interp" ]; then
    config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    # exp_root=${save_root}'/asva/asva_12_kf_interp/epoch=969-step=11640-uniform'
    exp_root=${save_root}'/asva/asva_12_kf_interp/open_domain-interp'
    checkpoint='/dockerx/share/Dynamicrafter_audio/main/save/asva_12_kf-interp/asva_12_kf-interp/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=969-step=11640.ckpt'
    FS=24
    video_length=48
fi

# panda:
# --config configs/inference_512_panda_12_keyframe.yaml \
# --exp_root save/panda/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_frameidx_position_embedding_swin \
# --checkpoint /dockerx/share/Dynamicrafter_audio/checkpoints/panda/epoch=0-step=3800-v5.ckpt \

# vgg_sound
# --config configs/inference_512_vgg_sound_12_keyframe.yaml \
# --exp_root save/vgg_sound/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf \
# --checkpoint /dockerx/share/Dynamicrafter_audio/checkpoints/vgg_sound/epoch=1-step=2950.ckpt \

    # --exp_root ${exp_root}_audio_${cfg_audio}_img_${cfg_img}_inpainting_step_${inpainting_end_step} \

run_asva() {
    local device_id=$1
    local cfg_audio=$2
    local cfg_img=$3
    local inpainting_end_step=$4
    CUDA_VISIBLE_DEVICES=$device_id python -W ignore scripts/evaluation/animation_gen.py \
    --config ${config} \
    --exp_root ${exp_root}_audio_${cfg_audio}_img_${cfg_img} \
    --checkpoint ${checkpoint} \
    --dataset AVSync15 \
    --height 320 \
    --width 512 \
    --video_fps ${FS} \
    --video_length ${video_length} \
    --num_clips_per_video 3 \
    --unconditional_guidance_scale 7.5 \
    --random_seed 0 \
    --ddim_steps 90 \
    --ddim_eta 1.0 \
    --text_input \
    --rank $device_id \
    --timestep_spacing 'uniform_trailing' --guidance_rescale 0.7 --perframe_ae \
    --cfg_audio $cfg_audio \
    --cfg_img $cfg_img \
    --multiple_cond_cfg \
    --interp 
    # --inpainting_end_step ${inpainting_end_step}
}


# for ((i=0; i<8; i++)); do
#     run_asva $i 2.0 2.0 &
#     sleep 1
# done

# for ((i=0; i<8; i++)); do
#     run_asva $i 4.0 2.0 0 &
#     sleep 1
# done

# wait 

for ((i=0; i<8; i++)); do
    run_asva $i 4.0 2.0 0
    sleep 1
done

# for ((i=0; i<8; i++)); do
#     run_asva $i 7.5 2.0 0
#     sleep 1
# done

# for ((i=0; i<8; i++)); do
#     run_asva $i 9.0 2.0 30 &
#     sleep 1
# done

# wait

# for ((i=0; i<8; i++)); do
#     run_asva $i 11.0 2.0 30 &
#     sleep 1
# done

# wait

# for ((i=0; i<8; i++)); do
#     run_asva $i 9.0 2.0 50 &
#     sleep 1
# done

# wait

# for ((i=0; i<8; i++)); do
#     run_asva $i 11.0 2.0 50 &
#     sleep 1
# done

# wait


# for ((i=0; i<8; i++)); do
#     run_asva $i 7.5 2.0 30 &
#     sleep 1
# done

# wait


# for ((i=0; i<8; i++)); do
#     run_asva $i 7.5 2.0 0 &
#     sleep 1
# done

# wait

# for ((i=0; i<8; i++)); do
#     run_asva $i 4.0 2.0 &
#     sleep 1
# done

# wait


# for ((i=0; i<8; i++)); do
#     run_asva $i 9.0 2.0 &
#     sleep 1
# done

# wait

# for ((i=0; i<8; i++)); do
#     # run_asva $i 11.0 2.0
#     run_asva $i 11.0 2.0 &
#     sleep 1
# done