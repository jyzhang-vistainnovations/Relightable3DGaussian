#!/bin/bash

scene="robot-4"
input_path="/c/Users/U/Documents/gs/Relightable/inputs/${scene}"
output_path_3dgs="/c/Users/U/Documents/gs/Relightable/outputs/${scene}/3dgs"
output_path_neilf="/c/Users/U/Documents/gs/Relightable/outputs/${scene}/neilfs"

python train.py \
    -s $input_path \
    -m $output_path_3dgs \
    --lambda_normal_render_depth 0.01 \
    --lambda_normal_smooth 0.01 \
    --lambda_mask_entropy 0.1 \
    --save_training_vis \
    --lambda_depth_var 1e-2

python train.py \
    -s $input_path \
    -m $output_path_neilf \
    -c $output_path_3dgs/chkpnt30000.pth \
    --save_training_vis \
    --position_lr_init 0.000016 \
    --position_lr_final 0.00000016 \
    --normal_lr 0.001 \
    --sh_lr 0.00025 \
    --opacity_lr 0.005 \
    --scaling_lr 0.0005 \
    --rotation_lr 0.0001 \
    --iterations 40000 \
    --lambda_base_color_smooth 0 \
    --lambda_roughness_smooth 0 \
    --lambda_light_smooth 0 \
    --lambda_light 0.01 \
    -t neilf --sample_num 64 \
    --save_training_vis_iteration 200 \
    --lambda_env_smooth 0.01
