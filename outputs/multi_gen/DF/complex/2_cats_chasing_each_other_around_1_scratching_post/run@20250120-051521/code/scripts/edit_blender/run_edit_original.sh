
# file="a plate of hotdog"
set -e

DEFAULT_FILE="objects/edit/lego/a_red_shovel_only.txt"
DEFAULT_GPU=2
DEFAULT_SCENE="lego"

gpu=${1:-$DEFAULT_GPU}
scene=${2:-$DEFAULT_SCENE}
file=${3:-$DEFAULT_FILE}


if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi



declare -A scene_dict
# scene_dict[chair]="outputs/nerf-blender-old/chair@20230919-020850/ckpts/last.ckpt"
# scene_dict[hotdog]="outputs/nerf-blender-old/hotdog@20230920-234745/ckpts/last.ckpt"
scene_dict[lego]="outputs/nerf-blender/lego@20240127-222414/ckpts/last.ckpt"
# scene_dict[ship]="outputs/nerf-blender-old/ship@20230919-024827/ckpts/last.ckpt"
scene_dict[materials]="outputs/nerf-blender/materials@20240130-220331/ckpts/last.ckpt"

echo  ${scene_dict[$scene]} 

base=$(basename "$file")
out_name=${base%.txt}
DEFAULT_CFG="dreamfusion-if-edit-blender-edit-original"

DIR=outputs/edit_blender/
name=${scene}/${out_name}

CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/$DEFAULT_CFG.yaml \
    --train --gpu 0  trainer.max_steps=32000 \
    resume=${scene_dict[$scene]}  system.gpt_file=$file \
    system.prompt_processor.prompt="run" exp_root_dir=$DIR name=$name \
    system.edit_original=false
