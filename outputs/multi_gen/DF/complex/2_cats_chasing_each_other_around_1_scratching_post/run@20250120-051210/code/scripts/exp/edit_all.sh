
#!/bin/bash

# Define common command parts
# script="run"
script="run_magic3d"

script_path="bash scripts/edit_blender/${script}.sh"
object_path="objects/edit"

# Define an array of specific command arguments (excluding GPU number)
    # "mic ${object_path}/mic/a_set_of_studio_speakers.txt"
    # "hotdog ${object_path}/hotdog/ketchup.txt"
    # "chair ${object_path}/chair/pineapple.txt"

commands=(
    "chair ${object_path}/chair/sitting_monkey.txt"
    "chair ${object_path}/chair/A_dancing_monkey_on_top_of_a_chair.txt"
    "chair ${object_path}/chair/pineapple.txt"
    "chair ${object_path}/chair/A_dancing_monkey.txt"
)

# Number of GPUs available
num_gpus=4

# Run each command on a different GPU
gpu=0
for cmd_args in "${commands[@]}"; do
    eval "$script_path $gpu $cmd_args &"
    gpu=$(( (gpu + 1) % num_gpus ))
done

# Wait for all background processes to finish
wait


# cmd="bash scripts/edit_blender/run_magic3d.sh 2 mic objects/edit/mic/a_set_of_stereo_speakers.txt & bash scripts/edit_blender/run_magic3d.sh 3 hotdog objects/edit/hotdog/ketchup.txt"
# eval $cmd
# wait 
# cmd="bash scripts/edit_blender/run_magic3d.sh 2 chair objects/edit/chair/A_dancing_monkey_on_top_of_a_chair.txt & bash scripts/edit_blender/run_magic3d.sh 3 material objects/edit/material/balls.txt"
# eval $cmd
# wait 
