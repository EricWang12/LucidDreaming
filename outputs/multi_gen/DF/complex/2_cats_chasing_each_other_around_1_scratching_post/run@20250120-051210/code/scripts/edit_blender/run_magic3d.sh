
# file="a plate of hotdog"
set -e

gpu=${1:-0}

DEFAULT_FILE="objects/edit/lego/an_apple_small_fig1.txt"
DEFAULT_GPU=1
DEFAULT_SCENE="tree"
DEFAULT_CFG="magic3d"

gpu=${1:-$DEFAULT_GPU}
scene=${2:-$DEFAULT_SCENE}
file=${3:-$DEFAULT_FILE}
anlge=${4:-0}
cfg=${5:-$DEFAULT_CFG}

if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi

if [[ $cfg != *.yaml ]]; then
    # If it doesn't, append '.yaml' to the filename
    # label="configs/DF-edit/${label}.yaml"
    cfg_file=configs/control/${cfg}.yaml
else    
    cfg_file=$cfg
fi


python scripts/control_inherit.py $cfg_file


declare -A scene_dict
scene_dict[chair]="outputs/nerf-blender-old/chair@20230919-020850/ckpts/last.ckpt"
scene_dict[hotdog]="outputs/nerf-blender-old/hotdog@20230920-234745/ckpts/last.ckpt"
scene_dict[lego]="outputs/nerf-blender-old/lego@20230919-023118/ckpts/last.ckpt"
scene_dict[ship]="outputs/nerf-blender-old/ship@20230919-024827/ckpts/last.ckpt"
scene_dict[material]="outputs/nerf-blender-old/materials@20230919-023713/ckpts/last.ckpt"
scene_dict[mic]="outputs/nerf-blender-old/mic@20230919-024315/ckpts/last.ckpt"
scene_dict[tree]="outputs/vanilla_generation/magic3d/Dan/a_tree/run@20240209-225408/ckpts/last.ckpt"

echo  ${scene_dict[$scene]} 

base=$(basename "$file")
out_name=${base%.txt}

DIR=outputs/edit_blender/
name=${scene}/magic3d/${out_name}

CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg_file  \
    --train --gpu 0  trainer.max_steps=40000  system.recon_loss_weight=0.3 \
    resume=${scene_dict[$scene]}  system.gpt_file=$file \
    system.prompt_processor.prompt="run" exp_root_dir=$DIR name=$name \
    system.prompt_processor.use_perp_neg=true  system.prompt_processor.rotation_angle=$anlge \
    system.background.random_aug=true \
    seed=8765
    # system.prompt_processor.front_threshold=60 system.prompt_processor.back_threshold=60 \
