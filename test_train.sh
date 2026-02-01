#!/bin/bash
# Training test script for SmolVLA with Wan2.2-TI2V-5B VAM
#
# This script tests training with the Wan2.2-TI2V-5B video generation model
# as the Video Action Model (VAM) backbone.
#
# Wan2.2-TI2V-5B specs:
# - 5B parameters (dense model)
# - 30 transformer layers
# - 3072 hidden dimension
# - 16x16x4 VAE compression ratio
# - Supports 720P @ 24fps video generation
#
# Memory estimate (~80GB GPU):
# - SmolVLM (0.5B): ~2 GB
# - Action Expert: ~1 GB
# - Wan2.2 VAE: ~1 GB
# - Wan2.2 Transformer (5B): ~10 GB (bf16)
# - Gradients (fine-tuning): ~20 GB
# - Activations: ~20 GB
# Total: ~55 GB (should fit on 80GB A100/H100)

rm -rf outputs/train/my_smolvla && \
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=julied1250/unscrew-bottle-merged \
  --batch_size=2 \
  --steps=5 \
  --policy.repo_id=brandonyang/temp \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=false \
  --policy.vam_hidden_dim=512 \
  --policy.vam_fusion_mode=sigmoid_gate \
  --policy.vam_num_tokens=64 \
  --policy.use_placeholder_vam=false \
  --policy.use_wan2_vam=true \
  --policy.wan2_model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers" \
  --policy.wan2_freeze=false \
  --policy.wan2_freeze_vae=true \
  --policy.wan2_freeze_text_embedder=true \
  --policy.wan2_trainable_layers=10
