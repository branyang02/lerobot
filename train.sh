#!/bin/bash
# Training script for SmolVLA with Wan2.2-TI2V-5B VAM
#
# Model: SmolVLM (0.5B) + Wan2.2-TI2V-5B (5B) + Action Expert
# Total: ~6B params, Trainable: ~1.7B params (with recommended freezing)
#
# Hardware: 4x A100 80GB GPUs
# Memory estimate per GPU: ~20-25GB (model + gradients + activations)

accelerate launch \
  --multi_gpu \
  --num_processes=4 \
  $(which lerobot-train) \
    --dataset.repo_id=julied1250/unscrew-bottle-merged \
    --output_dir=./outputs/vma-experiment \
    --job_name=smolvla-wan2-vam \
    --policy.repo_id=brandonyang/vma-experiment \
    --policy.path=lerobot/smolvla_base \
    --policy.device=cuda \
    --steps=30000 \
    --batch_size=64 \
    --wandb.enable=true \
    --save_freq=2000 \
    --log_freq=10 \
    --policy.vam_hidden_dim=512 \
    --policy.vam_fusion_mode=sigmoid_gate \
    --policy.vam_num_tokens=64 \
    --policy.use_placeholder_vam=false \
    --policy.use_wan2_vam=true \
    --policy.wan2_model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
    --policy.wan2_freeze=false \
    --policy.wan2_freeze_vae=true \
    --policy.wan2_freeze_text_embedder=true \
    --policy.wan2_trainable_layers=20
