
# file="a plate of hotdog"
set -e

gpu=${1:-0}

DEFAULT_FILE="objects/edit/lego/an_apple_small.txt"
DEFAULT_GPU=0
DEFAULT_SCENE="lego"

gpu=${1:-$DEFAULT_GPU}
scene=${2:-$DEFAULT_SCENE}
file=${3:-$DEFAULT_FILE}


if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi



declare -A scene_dict
scene_dict[chair]="outputs/nerf-blender/chair@20231006-002500/ckpts/last.ckpt"
scene_dict[hotdog]="outputs/nerf-blender/hotdog@20231006-004217/ckpts/last.ckpt"
scene_dict[lego]="outputs/nerf-blender/lego@20231006-004816/ckpts/last.ckpt"
scene_dict[ship]="outputs/nerf-blender/ship@20231006-010540/ckpts/last.ckpt"


echo  ${scene_dict[$scene]} 

base=$(basename "$file")
out_name=${base%.txt}
DEFAULT_CFG="DF-edit-blender/dreamfusion-if-edit-blender-material"

DIR=outputs/edit_blender/material/
name=${scene}/${out_name}

CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/$DEFAULT_CFG.yaml --train --gpu 0  trainer.max_steps=30000 \
    resume=${scene_dict[$scene]}  system.gpt_file=$file system.prompt_processor.use_perp_neg=true \
    system.prompt_processor.prompt="run" exp_root_dir=$DIR name=$name system.recon_loss_weight=0.5 \
