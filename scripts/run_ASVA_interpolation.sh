
#!/bin/bash

# DEVICE_ID=$1
DATASET=$1


# FS=6
# video_length=12

save_root='/dockerx/local/repo/DynamiCrafter/save'



run_asva() {
    local device_id=$1
    local cfg_audio=$2
    local cfg_img=$3
    local FS=$4
    local video_length=$5

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
    --multiple_cond_cfg 
}

run_inter() {
    local device_id=$1
    local cfg_audio=$2
    local cfg_img=$3
    local FS=$4
    local video_length=$5
    
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
    --inpainting
}



if [ $DATASET == "asva_48_inpainting" ]; then

    # 12 keyframes generation
    # config='configs/inference_512_asva_12_keyframe_new_add_idx.yaml'
    # exp_root=${save_root}'/asva/asva_12_keyframe_add_idx/epoch=309-step=3720-uniform'
    # checkpoint='checkpoints/asva_12_kf_add_idx/epoch=309-step=3720.ckpt'


    # FS=24
    # video_length=12

    
    # for ((i=0; i<8; i++)); do
    #     run_asva $i 7.5 2.0 $FS $video_length &
    #     sleep 1
    # done


    # interpolation
    config='configs/inference_512_asva_48_keyframe.yaml'
    exp_root=${save_root}'/asva/asva_48/epoch=579-step=13920-inpainting-2'
    checkpoint='checkpoints/asva_48_split_audio/epoch=579-step=13920.ckpt'
    
    FS=24
    video_length=48
    
    for ((i=0; i<8; i++)); do
        run_inter $i 7.5 2.0 ${FS} ${video_length} &
        sleep 1
    done

elif [ $DATASET == "asva_12_kf_freenoise" ]; then
    config='configs/inference_512_asva_48_keyframe.yaml'
    exp_root=${save_root}'/asva/asva_48/epoch=579-step=13920-inpainting'
    checkpoint='checkpoints/asva_48_split_audio/epoch=579-step=13920.ckpt'
fi
