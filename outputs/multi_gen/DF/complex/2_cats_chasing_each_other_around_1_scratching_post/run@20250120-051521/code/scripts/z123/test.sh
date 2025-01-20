DEFAULT_FILE="objects/z123/dog_catstat.txt"
DEFAULT_CFG="zero123"
DEFAULT_GPU=0
ckpt=${1:-""}
gpu=${2:-$DEFAULT_GPU}
file=${3:-$DEFAULT_FILE}
cfg=${4:-$DEFAULT_CFG}


cfg_file=configs/control/$cfg.yaml

python scripts/control_inherit.py $cfg_file

if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi
echo $file
echo $cfg


base=$(basename "$file")
out_name=${base%.txt}

DIR=outputs/multi_gen/0_TEST/

name=${cfg}/${out_name}

CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg_file --test --gpu 0 \
    system.background.random_aug=true system.gpt_file=$file \
    system.data.image_path="run"   exp_root_dir=$DIR name=$name resume=$ckpt \
    system.recon_loss_weight=0.3 system.data.random_camera.eval_camera_distance=10
    


    # system.prompt_processor.prompt="test" exp_root_dir=$DIR name=$name 
    # data.eval_camera_distance=4.5 \
    # data.eval_elevation_deg=0  data.eval_camera_distance=4


# file="an_apple_small"
# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg_file --train --gpu 0  system.gpt_file=objects/$file.txt resume=archive/outputs/nerf-blender/lego@20230919-023118/ckpts/last.ckpt system.prompt_processor.prompt="$file-$label"

# system.background.random_aug=true