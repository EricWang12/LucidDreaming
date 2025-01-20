ckpt=outputs/multi_gen/magic3d/good_box/Six_pumpkins_in_a_hexagon_shape,_with_a_scarecrow_in_the_center./run@20231102-224000/ckpts/last.ckpt
ckpt=outputs/multi_gen/magic3d/good_box/Four_apples_arranged_in_a_square_with_a_pear_in_the_middle./run@20231102-091242/ckpts/last.ckpt
# Extract the prompt
file=objects/multi_gen/complex/Four_apples_arranged_in_a_square_with_a_pear_in_the_middle..txt
prompt=$(echo $ckpt | awk -F '/' '{print $5}' | sed 's/_/ /g')

CUDA_VISIBLE_DEVICES=0 python launch.py --config configs/control/magic3d-refine-sd.yaml --train --gpu 0 \
    system.prompt_processor.prompt="$prompt" system.gpt_file=$file  system.renderer.gpt_file=$file  \
    system.geometry_convert_from=$ckpt \