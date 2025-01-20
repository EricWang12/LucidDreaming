set -e

gpu=$1
label=$2
# prompt=${3:-"A green car parking on the left of a blue truck, with a red air balloon in the sky"}
ckpt=${3:-"outputs/vanilla_generation/DF/six_apples_arranged_in_a_three_by_two_grid/run@20231017-120452/ckpts/last.ckpt"}
cls=${4:-"hard"}
prompt="run"
declare -A cfg_dict
cfg_dict[PD]="configs/prolificdreamer.yaml"
cfg_dict[PD-p]="configs/prolificdreamer-test.yaml"
cfg_dict[DF]="configs/dreamfusion-if-test.yaml"
cfg_dict[magic3d]="configs/magic3d-coarse-if.yaml"

prompt=${prompt// /_}

if [[ ! -v cfg_dict[$label] ]]; then
    echo "$label doesn't exists!!!"
    exit 1
fi


DIR=outputs/vanilla_generation/0_TEST/
name=${label}/${cls}/${prompt// /_}


echo  ${cfg_dict[$label]} 

echo $DIR$name
CUDA_VISIBLE_DEVICES=$gpu python launch.py --config ${cfg_dict[$label]} --test --gpu 0 \
    system.prompt_processor.prompt="${prompt}" exp_root_dir=$DIR name=$name tag="run" \
    resume=$ckpt #system.background_type="solid-color-background" system.background.color="0.0,0.0,0.0" \