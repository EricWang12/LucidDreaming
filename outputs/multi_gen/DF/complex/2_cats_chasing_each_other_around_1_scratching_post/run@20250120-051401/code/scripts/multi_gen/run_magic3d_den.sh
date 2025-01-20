cfg_file=configs/control/magic3d-den.yaml
ckpt=outputs/Dan/tree/run@20240210-181242/ckpts/last.ckpt
gpu=${1:-0}
weight=${2:-0.1}
python scripts/control_inherit.py $cfg_file


CUDA_VISIBLE_DEVICES=1 python launch.py --config $cfg_file  \
    --train --gpu 0  trainer.max_steps=30000  system.recon_loss_weight=$weight \
    resume=$ckpt  system.gpt_file=objects/Dan/tree_snake_one.txt \
    system.prompt_processor.prompt="run" exp_root_dir=outputs/multi_gen/magic3d/Dan name=tree_density \
    system.prompt_processor.use_perp_neg=true   \
    system.background.random_aug=true \
    seed=8765