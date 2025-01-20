set -e

DEFAULT_FILE="an_apple_corner"
DEFAULT_CFG="configs/DF-edit/DF.yaml"
DEFAULT_GPU=0
gpu=${1:-$DEFAULT_GPU}
file=${2:-$DEFAULT_FILE}
cfg=${3:-$DEFAULT_CFG}


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



CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg --train --gpu 0 \
    system.background.random_aug=true system.gpt_file=$file \
    system.prompt_processor.prompt="run" exp_root_dir=$DIR name=$name \
    system.global_guidance_start=4000  system.recon_loss_weight=0.1