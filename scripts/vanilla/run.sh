set -e

gpu=$1
label=$2
prompt=${3:-"A green car parking on the left of a blue truck, with a red air balloon in the sky"}
cls=${4:-"hard"}

declare -A cfg_dict
cfg_dict[PD]="configs/prolificdreamer.yaml"
cfg_dict[PD-p]="configs/prolificdreamer-patch.yaml"
cfg_dict[DF]="configs/dreamfusion-if.yaml"
cfg_dict[magic3d]="configs/magic3d-coarse-if.yaml"

prompt=${prompt// /_}

if [[ ! -v cfg_dict[$label] ]]; then
    echo "$label doesn't exists!!!"
    exit 1
fi


DIR=outputs/vanilla_generation/
name=${label}/${cls}/${prompt// /_}


echo  ${cfg_dict[$label]} 

echo $DIR$name
CUDA_VISIBLE_DEVICES=$gpu python launch.py --config ${cfg_dict[$label]} --train --gpu 0 system.prompt_processor.prompt="${prompt}" \
    exp_root_dir=$DIR name=$name tag="run" system.geometry.density_bias="blob_box" system.geometry.gpt_file="objects/interact/pig_with_6_legs.txt"