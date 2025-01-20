set -e

DEFAULT_FILE="an_apple_corner"
DEFAULT_CFG="configs/DF-edit/DF.yaml"
DEFAULT_GPU=0
DEFAULT_CKPT=""
gpu=${1:-$DEFAULT_GPU}
file=${2:-$DEFAULT_FILE}
ckpt=${3:-$DEFAULT_CKPT}
cfg=${4:-$DEFAULT_CFG}


if [[ $file != *.txt ]]; then
    file=objects/$file.txt
fi
echo $file
echo $cfg




base=$(basename "$file")
out_name=${base%.txt}

base_cfg=$(basename "$cfg")
out_cfg=${base_cfg%.yaml}

DIR=outputs/multi_gen/0_TEST/

name=${out_cfg}/${out_name}



CUDA_VISIBLE_DEVICES=$gpu python launch.py --config $cfg --test --gpu 0 \
    system.background.random_aug=true resume=$ckpt  \
    system.gpt_file=$file system.prompt_processor.prompt="run" exp_root_dir=$DIR name=$name