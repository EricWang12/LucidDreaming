DEFAULT_GPU=0
gpu=${1:-$DEFAULT_GPU}
label=${2:"PD-p"}
prompt=${3:-"A green car parking on the left of a blue truck, with a red air balloon in the sky"}

# declare -A cfg_dict
# cfg_dict[PD]="configs/prolificdreamer.yaml"
# cfg_dict[PD-p]="configs/prolificdreamer-patch.yaml"
# cfg_dict[DF]="configs/dreamfusion-if.yaml"
# cfg_dict[magic3d]="configs/magic3d-coarse-if.yaml"


# if [[ ! -v cfg_dict[$label] ]]; then
#     echo "$label doesn't exists!!!"
#     exit 1
# fi

DIR=outputs/jinqi/
# name=${label}/${prompt// /_}

echo $gpu
echo $label
echo  ${cfg_dict[$label]} 


# Define the list of string arguments
string_list=("Icthyophaga Leucogaster" "Icthyophaga Leucogaster, also known as white-bellied sea eagle, white head, yellow legs, black claws, hooked bill")

# Loop through the string list and execute run.sh with each string as argument
for string in "${string_list[@]}"; do

    CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/prolificdreamer-patch-jinqi.yaml --train --gpu 0 system.prompt_processor.prompt="${string}" exp_root_dir=$DIR
done