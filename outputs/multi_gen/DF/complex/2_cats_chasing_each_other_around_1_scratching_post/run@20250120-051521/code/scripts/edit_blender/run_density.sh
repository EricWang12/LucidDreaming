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
echo $file
echo $cfg



declare -A scene_dict
scene_dict[chair]="outputs/nerf-blender-old/chair@20230919-020850/ckpts/last.ckpt"
scene_dict[hotdog]="outputs/nerf-blender-old/hotdog@20230920-234745/ckpts/last.ckpt"
scene_dict[lego]="outputs/nerf-blender-old/lego@20230919-023118/ckpts/last.ckpt"
scene_dict[ship]="outputs/nerf-blender-old/ship@20230919-024827/ckpts/last.ckpt"


echo  ${scene_dict[$scene]} 



base=$(basename "$file")
out_name=${base%.txt}

base_cfg=$(basename "$cfg")
out_cfg=${base_cfg%.yaml}


DIR=outputs/edit_blender/density/
name=${scene}/${out_name}

echo $name


CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/dreamfusion-if-edit-blender-den.yaml \
    --train --gpu 0  trainer.max_steps=30000 system.global_guidance_start=3000  \
    resume=${scene_dict[$scene]}  system.gpt_file=$file \
    system.prompt_processor.prompt="run" exp_root_dir=$DIR   name=$name


# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/dreamfusion-if-edit-blender-test.yaml --test --gpu 0 \
#     resume=outputs/dreamfusion-if-edit/pineapple@20231011-154739/ckpts/last.ckpt\
#     system.gpt_file=$file system.prompt_processor.prompt="run" exp_root_dir=$DIR name=$name