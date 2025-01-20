#!/bin/bash

DEFAULT_FILE="an_apple_corner"
DEFAULT_CFG="control-wrapper"
DEFAULT_GPU=0
gpu=${1:-$DEFAULT_GPU}
file=${2:-$DEFAULT_FILE}
label=${3:-$DEFAULT_CFG}
ckpt=${4:-""}

cfg_file=configs/control/test/$label.yaml

python scripts/control_inherit.py $cfg_file

if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi
echo $file
echo $label


base=$(basename "$file")
out_name=${base%.txt}

DIR=outputs/multi_gen/0_TEST/

name=${label}/${out_name}

CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg_file --test --gpu 0 system.gpt_file=$file \
    system.prompt_processor.prompt="test" exp_root_dir=$DIR name=$name resume=$ckpt \
    system.recon_loss_weight=1.0 data.eval_camera_distance=4.5 \
    # data.eval_elevation_deg=0  data.eval_camera_distance=4


# file="an_apple_small"
# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg_file --train --gpu 0  system.gpt_file=objects/$file.txt resume=archive/outputs/nerf-blender/lego@20230919-023118/ckpts/last.ckpt system.prompt_processor.prompt="$file-$label"

# system.background.random_aug=true