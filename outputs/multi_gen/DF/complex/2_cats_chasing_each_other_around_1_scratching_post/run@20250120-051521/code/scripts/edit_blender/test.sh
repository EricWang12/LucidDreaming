
# file="a plate of hotdog"
set -e

gpu=${1:-0}

DEFAULT_FILE="objects/edit/lego/an_apple_small.txt"
DEFAULT_GPU=0
DEFAULT_SCENE="lego"
DEFAULT_CKPT=""
DEFAULT_CFG="magic3d-blender"

gpu=${1:-$DEFAULT_GPU}
scene=${2:-$DEFAULT_SCENE}
file=${3:-$DEFAULT_FILE}
ckpt=${4:-$DEFAULT_CKPT}
cfg=${5:-$DEFAULT_CFG}


if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi



declare -A scene_dict
scene_dict[chair]="outputs/nerf-blender-old/chair@20230919-020850/ckpts/last.ckpt"
scene_dict[hotdog]="outputs/nerf-blender-old/hotdog@20230920-234745/ckpts/last.ckpt"
# scene_dict[lego]="outputs/nerf-blender-old/lego@20230919-023118/ckpts/last.ckpt"
scene_dict[ship]="outputs/nerf-blender-old/ship@20230919-024827/ckpts/last.ckpt"
scene_dict[materials]="outputs/nerf-blender/materials@20240130-220331/ckpts/last.ckpt"
scene_dict[lego]="outputs/nerf-blender/lego@20240127-222414/ckpts/last.ckpt"


echo  ${scene_dict[$scene]} 

base=$(basename "$file")
out_name=${base%.txt}
DEFAULT_CFG="dreamfusion-if-edit-blender"

DIR=outputs/edit_blender/0_TEST/
name=${scene}/${out_name}
# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/dreamfusion-if-edit-blender-test.yaml --test --gpu 0  trainer.max_steps=30000 \

CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/control/test/$cfg.yaml  --test --gpu 0  trainer.max_steps=30000 \
    resume=$ckpt system.gpt_file=$file \
    system.prompt_processor.prompt="run" exp_root_dir=$DIR name=$name \
    # data.eval_elevation_deg=35 #system.scene_edit=false  \
