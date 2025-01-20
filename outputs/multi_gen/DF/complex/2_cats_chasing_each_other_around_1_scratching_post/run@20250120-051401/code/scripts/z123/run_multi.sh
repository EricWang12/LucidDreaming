#!/bin/bash

DEFAULT_FILE="objects/z123/dog_catstat.txt"
DEFAULT_CFG="zero123"
DEFAULT_GPU=0
gpu=${1:-$DEFAULT_GPU}
file=${2:-$DEFAULT_FILE}
cfg=${3:-$DEFAULT_CFG}



if [[ $cfg != *.yaml ]]; then
    # If it doesn't, append '.yaml' to the filename
    cfg_file=configs/control/$cfg.yaml
    else
    cfg_file=$cfg
fi

python scripts/control_inherit.py $cfg_file

if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi
echo $file
echo $cfg


base=$(basename "$file")
out_name=${base%.txt}

base_cfg=$(basename "$cfg")
out_cfg=${base_cfg%.yaml}

cls=$(basename $(dirname  $file))

DIR=outputs/multi_gen/

name=${out_cfg}/${cls}/${out_name}

echo $DIR$name

CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg_file --train --gpu 0  \
    system.background.random_aug=true system.gpt_file=$file \
    system.data.image_path="run"   exp_root_dir=$DIR name=$name \
    system.recon_loss_weight=1.0
    # system.data.default_camera_distance=3.5 \
    # system.data.random_camera.eval_camera_distance=6.5\
    # system.prompt_processor.use_perp_neg=true \

# file="an_apple_small"
# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg_file --train --gpu 0  system.gpt_file=objects/$file.txt resume=archive/outputs/nerf-blender/lego@20230919-023118/ckpts/last.ckpt system.prompt_processor.prompt="$file-$cfg"

# system.background.random_aug=true