#!/bin/bash
# Inference test script for SmolVLA with Wan2.2-TI2V-5B VAM
#
# This script tests inference with the Wan2.2-TI2V-5B video generation model
# as the Video Action Model (VAM) backbone.
#
# Requires: A checkpoint from test_train.sh

python -c "
import torch
from pathlib import Path

# Find the latest checkpoint
checkpoint_dir = Path('outputs/train/my_smolvla/checkpoints')
if not checkpoint_dir.exists():
    print('Error: No checkpoint found. Run test_train.sh first.')
    exit(1)

checkpoints = sorted(checkpoint_dir.iterdir())
if not checkpoints:
    print('Error: No checkpoint found. Run test_train.sh first.')
    exit(1)

policy_path = checkpoints[-1] / 'pretrained_model'
print(f'Loading checkpoint: {policy_path}')

# Load the policy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicyi 

policy = SmolVLAPolicy.from_pretrained(policy_path)

# Verify Wan2.2 VAM configuration
print(f'\nPolicy configuration:')
print(f'  - VAM hidden dim: {policy.config.vam_hidden_dim}')
print(f'  - VAM fusion mode: {policy.config.vam_fusion_mode}')
print(f'  - VAM num tokens: {policy.config.vam_num_tokens}')
print(f'  - Use Wan2.2 VAM: {policy.config.use_wan2_vam}')
print(f'  - Wan2.2 model ID: {policy.config.wan2_model_id}')
print(f'  - Wan2.2 freeze: {policy.config.wan2_freeze}')

# Check if VAM is enabled
print(f'\nModel status:')
print(f'  - VAM enabled: {policy.model.vlm_with_expert.vam_enabled}')
print(f'  - Use Wan2.2 VAM: {policy.model.vlm_with_expert.use_wan2_vam}')

policy = policy.to('cuda')
policy.eval()

# Create fake batch for inference test
batch_size = 2
img_size = 512
state_dim = 6
chunk_size = 50

# Fake inputs matching the training data structure
fake_batch = {
    'observation.images.camera1': torch.rand(batch_size, 3, img_size, img_size, device='cuda'),
    'observation.images.camera2': torch.rand(batch_size, 3, img_size, img_size, device='cuda'),
    'observation.images.camera3': torch.rand(batch_size, 3, img_size, img_size, device='cuda'),
    'observation.state': torch.rand(batch_size, state_dim, device='cuda'),
    'observation.language.tokens': torch.randint(0, 1000, (batch_size, 48), device='cuda'),
    'observation.language.attention_mask': torch.ones(batch_size, 48, device='cuda', dtype=torch.bool),
}

print(f'\nRunning inference with Wan2.2 VAM...')
print(f'  - Batch size: {batch_size}')
print(f'  - Image size: {img_size}x{img_size}')
print(f'  - State dim: {state_dim}')

with torch.no_grad():
    actions = policy.predict_action_chunk(fake_batch)

print(f'\nInference successful!')
print(f'  - Output actions shape: {actions.shape}')
print(f'  - Expected shape: [{batch_size}, {chunk_size}, action_dim]')
print(f'\nAll tests passed!')
"
