set -e

gpu=${1:-0}
prompt=$2
label=$3
edit=$4

DIRECTORY="load/nerf_synthetic"

# for subdir in "$DIRECTORY"/*/; do
#     # Extract just the name of the subdir without the path
#     subdir_name=$(basename "$subdir")
#     # python launch.py --config configs/dreamfusion-if.yaml --test --gpu 2 resume=outputs/dreamfusion-if/a_delicious_hamburger@20230913-131048/ckpts/last.ckpt system.prompt_processor.prompt="an apple"
#     CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/nerf-blender-ts.yaml --train --gpu 0  trainer.max_steps=20000 data.scene=$subdir_name tag=$subdir_name
# done
CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/nerf-blender-ts-nobias.yaml --train --gpu 0  trainer.max_steps=20000 data.scene=materials tag=materials

# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/nerf-blender-ts-nobias.yaml --train --gpu 0  trainer.max_steps=20000 data.scene=lego tag=lego

# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/nerf-colmap-ts.yaml --train --gpu 0  trainer.max_steps=20000 data.scene=garden tag=garden
# CUDA_VISIBLE_DEVICES=$gpu python launch.py --config configs/nerf-colmap-horse-test.yaml --test --gpu 0  trainer.max_steps=20000  tag=horse resume=/home/eric/workspace/3D/threestudio/outputs/nerf-colmap/horse@20231004-211453/ckpts/last.ckpt
