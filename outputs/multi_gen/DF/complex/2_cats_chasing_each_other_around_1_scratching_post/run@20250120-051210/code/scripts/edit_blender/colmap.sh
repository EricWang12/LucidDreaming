set -e

gpu=${1:-1}
file="edit/garden/an_apple_small"
label='apple'



CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/dreamfusion-if-edit-col.yaml --train --gpu 0  trainer.max_steps=30000 \
    resume=outputs/nerf-colmap/garden@20231011-233912/ckpts/last.ckpt system.gpt_file=objects/$file.txt system.prompt_processor.prompt="$label"
