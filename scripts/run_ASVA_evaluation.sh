
EXPSET=$1

save_root='/dockerx/local/repo/DynamiCrafter/save'


if [ $EXPSET == "asva" ]; then

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

    
elif [ $EXPSET == "dynamicrafter" ]; then
    config='configs/inference_512_v1.0.yaml'
    exp_root=${save_root}'/asva/asva_dynamicrafter'
    checkpoint='checkpoints/dynamicrafter_512_v1/model.ckpt'
    FS=6
    video_length=12


elif [ $EXPSET == "asva_48" ]; then
    config='configs/inference_512_asva_48_keyframe.yaml'
    exp_root=${save_root}'/asva/asva_48/epoch=579-step=13920'
    # checkpoint='checkpoints/asva_48_split_audio/epoch=179-step=4320.ckpt'
    checkpoint='checkpoints/asva_48_split_audio/epoch=579-step=13920.ckpt'
    FS=24
    video_length=48


elif [ $EXPSET == "asva_12_kf" ]; then
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

elif [ $EXPSET == "asva_12_kf_interp" ]; then
    config='configs/inference_512_asva_12_keyframe_kf_freenoise.yaml'
    exp_root=${save_root}'/asva/asva_12_kf_interp/reproduce_interp'

    # exp_root=${save_root}'/asva/asva_12_kf_interp/epoch=969-step=11640-uniform'
    # exp_root=${save_root}'/asva/asva_12_kf_interp/open_domain-interp'
    # exp_root=${save_root}'/asva/asva_12_kf_interp_no_idx/epoch=849-step=10200-interp'
    # 
    # checkpoint='/dockerx/groups/asva_12_kf-interp/asva_12_kf-interp/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=969-step=11640.ckpt'
    checkpoint='/dockerx/share/KeyVID/main/save/asva_12_kf-interp-more/training_512_avsyn_qformer_12_keyframe_framequery_60audio_random_frames_fps_cf/checkpoints/epoch=1479-step=17760.ckpt'

    FS=24
    video_length=48

    keyframe_gen_dir='/dockerx/groups/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_audio_7.5_img_2.0/samples'
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
    local cfg_audio_2=$5
    CUDA_VISIBLE_DEVICES=$device_id python -W ignore scripts/evaluation/animation_gen.py \
    --config ${config} \
    --exp_root ${exp_root}_audio_${cfg_audio}_img_${cfg_img}_kf_${cfg_audio_2} \
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
    --interp \
    --keyframe_gen_dir $keyframe_gen_dir
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
    keyframe_gen_dir='/dockerx/groups/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_audio_7.5_img_2.0/samples'
    run_asva $i 7.5 2.0 0 7.5 &
    sleep 1
done

wait

for ((i=0; i<8; i++)); do
    keyframe_gen_dir='/dockerx/groups/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_audio_9.0_img_2.0/samples'
    run_asva $i 9.0 2.0 0 9.0 &
    sleep 1
done

wait

for ((i=0; i<8; i++)); do
    keyframe_gen_dir='/dockerx/groups/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_audio_7.5_img_2.0/samples'
    run_asva $i 11.0 2.0 0 7.5 &
    sleep 1
done

wait

for ((i=0; i<8; i++)); do
    keyframe_gen_dir='/dockerx/groups/asva_12_kf_add_idx_add_fps/epoch=1339-step=16080-kf_audio_9.0_img_2.0/samples'
    run_asva $i 11.0 2.0 0 9.0 &
    sleep 1
done

wait

# for ((i=0; i<8; i++)); do
#     run_asva $i 7.5 2.0 0 &
#     sleep 1
# done

# wait

# for ((i=0; i<8; i++)); do
#     run_asva $i 8 2.0 0 &
#     sleep 1
# done

# wait
