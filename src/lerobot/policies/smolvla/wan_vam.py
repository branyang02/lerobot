# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Wan2.2-TI2V-5B Video Action Model (VAM) for SmolVLA.

This module wraps the Wan2.2 video generation model to extract per-layer
hidden states for cross-attention with the action expert.
"""

import torch
import torch.nn as nn


class Wan22VAM(nn.Module):
    """
    Wan2.2-TI2V-5B based Video Action Model.

    Uses the full WanTransformer3DModel (30 layers, 3072 hidden dim, 5B params)
    from Wan-AI/Wan2.2-TI2V-5B to extract per-layer hidden states for
    cross-attention with the action expert.

    Model specs (Wan2.2-TI2V-5B):
    - 5B parameters (dense, not MoE)
    - 30 transformer layers
    - 3072 hidden dimension (48 heads × 64 head_dim)
    - Wan2.2-VAE with 16×16×4 compression ratio
    - Supports 720P @ 24fps video generation
    - Can run on consumer GPUs (4090 with offloading)

    Architecture:
    - Input: Camera images [B, num_cameras, C, H, W]
    - Encode with Wan2.2 VAE to latents (16×16×4 compression)
    - Process through WanTransformer3DModel with forward hooks
    - Extract per-layer hidden states and project to expert dimension

    Dimension flow:
    - Input images: [B, 3, 3, 512, 512] (3 cameras stacked as video frames)
    - VAE latents: [B, 16, T/4, H/16, W/16] = [B, 16, 1, 32, 32] for single frame
    - Patchified: [B, num_patches, 3072]
    - Per-layer hidden: [B, num_patches, 3072]
    - Token compressed: [B, vam_num_tokens, 3072]
    - Final projection: [B, vam_num_tokens, vam_hidden_dim]
    """

    def __init__(
        self,
        model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        num_expert_layers: int = 16,
        vam_hidden_dim: int = 512,
        vam_num_tokens: int = 64,
        freeze_backbone: bool = False,
        freeze_vae: bool = True,
        freeze_text_embedder: bool = True,
        trainable_layers: int = -1,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Initialize Wan2.2-TI2V-5B VAM.

        Args:
            model_id: HuggingFace model ID for Wan2.2-TI2V-5B (diffusers format).
                      Default: "Wan-AI/Wan2.2-TI2V-5B-Diffusers" (5B params, 30 layers, 3072 hidden).
            num_expert_layers: Number of expert layers to map to (default 16).
            vam_hidden_dim: Output hidden dimension (should match expert's K/V input, default 512).
            vam_num_tokens: Number of output tokens after compression (default 64).
            freeze_backbone: Whether to freeze entire Wan2.2 backbone (overrides other freeze options).
            freeze_vae: Whether to freeze VAE encoder (recommended=True, saves 700M params).
            freeze_text_embedder: Whether to freeze text/time embedder (recommended=True, we use dummy text).
            trainable_layers: Number of transformer layers to train from the END.
                             -1 = all layers trainable
                             e.g., 10 = train last 10 layers (20-29), freeze first 20 (0-19)
            dtype: Data type for transformer (typically bfloat16 for memory efficiency).
        """
        super().__init__()

        self.model_id = model_id
        self.num_expert_layers = num_expert_layers
        self.vam_hidden_dim = vam_hidden_dim
        self.vam_num_tokens = vam_num_tokens
        self.dtype = dtype
        self.freeze_backbone = freeze_backbone
        self.freeze_vae = freeze_vae if not freeze_backbone else True
        self.freeze_text_embedder = freeze_text_embedder if not freeze_backbone else True
        self.trainable_layers = trainable_layers if not freeze_backbone else 0

        # These will be set after loading the model
        self.wan2_hidden_dim = None
        self.wan2_num_layers = None
        self.layer_indices = None

        # Storage for hooked hidden states
        self._layer_outputs = {}
        self._hook_handles = []

        # Components (loaded lazily or explicitly)
        self.vae = None
        self.transformer = None
        self.token_compressor = None
        self.projections = None

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load Wan2.2-TI2V-5B VAE and transformer from HuggingFace."""
        try:
            from diffusers import AutoencoderKLWan, WanTransformer3DModel
        except ImportError as e:
            raise ImportError(
                "diffusers with Wan2.2 support is required. "
                "Install with: pip install diffusers>=0.32.0"
            ) from e

        print(f"Loading Wan2.2-TI2V-5B VAM from {self.model_id}...")
        print("  Model specs: 5B params, 30 layers, 3072 hidden dim")
        print("  VAE compression: 16×16×4 (spatial×temporal)")

        # Load Wan2.2 VAE for image encoding (high-compression VAE)
        print("  Loading Wan2.2-VAE (16×16×4 compression)...")
        self.vae = AutoencoderKLWan.from_pretrained(
            self.model_id,
            subfolder="vae",
            torch_dtype=torch.float32,  # VAE needs fp32 for numerical stability
        )
        vae_params = sum(p.numel() for p in self.vae.parameters()) / 1e6
        print(f"    VAE parameters: {vae_params:.1f}M")

        # Load Wan2.2 transformer (5B parameters)
        print("  Loading WanTransformer3DModel (5B)...")
        self.transformer = WanTransformer3DModel.from_pretrained(
            self.model_id,
            subfolder="transformer",
            torch_dtype=self.dtype,
        )
        transformer_params = sum(p.numel() for p in self.transformer.parameters()) / 1e9
        print(f"    Transformer parameters: {transformer_params:.2f}B")

        # Get model dimensions from config
        # Wan2.2-TI2V-5B: 48 heads × 64 head_dim = 3072 hidden dim
        self.wan2_hidden_dim = self.transformer.config.num_attention_heads * self.transformer.config.attention_head_dim
        self.wan2_num_layers = self.transformer.config.num_layers

        print(f"  Wan2.2 architecture:")
        print(f"    Hidden dim: {self.wan2_hidden_dim}")
        print(f"    Num layers: {self.wan2_num_layers}")
        print(f"    Num attention heads: {self.transformer.config.num_attention_heads}")
        print(f"    Head dim: {self.transformer.config.attention_head_dim}")

        # Layer mapping: select num_expert_layers from wan2_num_layers
        # For 30 -> 16: sample roughly every 2nd layer
        self.layer_indices = self._compute_layer_indices()
        print(f"  Layer mapping ({self.wan2_num_layers} -> {self.num_expert_layers}):")
        print(f"    Sampled layer indices: {self.layer_indices}")

        # Register hooks to capture hidden states
        self._register_hooks()

        # Initialize compression and projection layers
        self._init_projection_layers()

        # Apply freezing based on configuration
        self._apply_freezing()

        total_params = (vae_params * 1e6 + transformer_params * 1e9) / 1e9
        trainable = self.get_trainable_params()
        trainable_total = trainable["total"] / 1e9
        print(f"  Wan2.2 VAM loaded! Total: {total_params:.2f}B, Trainable: {trainable_total:.2f}B")

    def _apply_freezing(self):
        """Apply fine-grained freezing to Wan2.2 components."""
        print("  Freezing configuration:")

        # 1. Freeze entire backbone if requested
        if self.freeze_backbone:
            print("    - Entire backbone: FROZEN")
            for p in self.vae.parameters():
                p.requires_grad = False
            for p in self.transformer.parameters():
                p.requires_grad = False
            return

        # 2. Freeze VAE (recommended - it's just for compression)
        if self.freeze_vae:
            print("    - VAE encoder: FROZEN (saves 700M params)")
            for p in self.vae.parameters():
                p.requires_grad = False
        else:
            print("    - VAE encoder: TRAINABLE")

        # 3. Freeze text/time embedder (recommended - we use dummy text)
        if self.freeze_text_embedder:
            print("    - Text/time embedder: FROZEN")
            if hasattr(self.transformer, 'condition_embedder'):
                for p in self.transformer.condition_embedder.parameters():
                    p.requires_grad = False
        else:
            print("    - Text/time embedder: TRAINABLE")

        # 4. Freeze early transformer layers (train only last N layers)
        if self.trainable_layers > 0 and self.trainable_layers < self.wan2_num_layers:
            freeze_until = self.wan2_num_layers - self.trainable_layers
            print(f"    - Transformer layers 0-{freeze_until-1}: FROZEN")
            print(f"    - Transformer layers {freeze_until}-{self.wan2_num_layers-1}: TRAINABLE")

            for i, block in enumerate(self.transformer.blocks):
                if i < freeze_until:
                    for p in block.parameters():
                        p.requires_grad = False

            # Also freeze patch_embedding, rope, and other early components
            if hasattr(self.transformer, 'patch_embedding'):
                for p in self.transformer.patch_embedding.parameters():
                    p.requires_grad = False
            if hasattr(self.transformer, 'rope'):
                for p in self.transformer.rope.parameters():
                    p.requires_grad = False
        elif self.trainable_layers == -1:
            print(f"    - All {self.wan2_num_layers} transformer layers: TRAINABLE")
        else:
            print(f"    - All {self.wan2_num_layers} transformer layers: TRAINABLE")

        # Always keep output projection trainable if not freezing backbone
        if hasattr(self.transformer, 'proj_out'):
            for p in self.transformer.proj_out.parameters():
                p.requires_grad = True
        if hasattr(self.transformer, 'norm_out'):
            for p in self.transformer.norm_out.parameters():
                p.requires_grad = True

    def _compute_layer_indices(self) -> list[int]:
        """Map wan2_num_layers to num_expert_layers via even sampling."""
        if self.wan2_num_layers <= self.num_expert_layers:
            # Use all layers if fewer than expert layers
            return list(range(self.wan2_num_layers))

        # Evenly sample layers
        step = self.wan2_num_layers / self.num_expert_layers
        indices = [min(int(i * step), self.wan2_num_layers - 1) for i in range(self.num_expert_layers)]
        return indices

    def _register_hooks(self):
        """Register forward hooks to capture intermediate hidden states."""
        # Clear existing hooks
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        def make_hook(layer_idx):
            def hook(module, input, output):
                # WanTransformerBlock returns hidden_states directly
                self._layer_outputs[layer_idx] = output
            return hook

        # Register hooks on selected transformer blocks
        # WanTransformer3DModel uses 'blocks' not 'transformer_blocks'
        for idx in self.layer_indices:
            block = self.transformer.blocks[idx]
            handle = block.register_forward_hook(make_hook(idx))
            self._hook_handles.append(handle)

    def _init_projection_layers(self):
        """Initialize token compression and projection layers."""
        # Token compression: estimate initial token count based on typical input
        # For 512x512 images with 16x16 spatial compression and patchify:
        # tokens = (512/16/2) * (512/16/2) * num_frames = 16 * 16 * 3 = 768 (approx)
        # We'll use a default and resize if needed
        self._default_num_tokens = 1024  # Conservative estimate
        self.token_compressor = nn.Linear(self._default_num_tokens, self.vam_num_tokens)
        # Initialize with small values for stable training
        nn.init.normal_(self.token_compressor.weight, std=0.02)
        nn.init.zeros_(self.token_compressor.bias)

        # Projection layers: wan2_hidden_dim -> vam_hidden_dim (one per expert layer)
        self.projections = nn.ModuleList([
            nn.Linear(self.wan2_hidden_dim, self.vam_hidden_dim)
            for _ in range(self.num_expert_layers)
        ])

    def _ensure_token_compressor(self, num_tokens: int, device: torch.device):
        """Resize token compressor if needed for different input sizes."""
        if self.token_compressor.in_features != num_tokens:
            print(f"  Resizing token_compressor: {self.token_compressor.in_features} -> {num_tokens}")
            # Create new compressor with correct size
            old_compressor = self.token_compressor
            self.token_compressor = nn.Linear(num_tokens, self.vam_num_tokens, device=device)
            # Initialize with small values for stable training
            nn.init.normal_(self.token_compressor.weight, std=0.02)
            nn.init.zeros_(self.token_compressor.bias)
            # Copy dtype from old compressor
            self.token_compressor = self.token_compressor.to(dtype=old_compressor.weight.dtype)

    def forward(
        self,
        images: torch.Tensor,
        timestep: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        """
        Process camera images through Wan2.2 and return per-layer hidden states.

        Args:
            images: Camera images [B, num_cameras, C, H, W] or [B, C, num_cameras, H, W]
            timestep: Diffusion timestep (default: 0 for clean images / feature extraction)

        Returns:
            List of hidden states [num_expert_layers x [B, vam_num_tokens, vam_hidden_dim]]
        """
        batch_size = images.shape[0]
        device = images.device

        # Handle different input formats
        if images.dim() == 5:
            if images.shape[1] == 3 and images.shape[2] != 3:
                # [B, C, num_cameras, H, W] -> already correct format [B, C, T, H, W]
                pass
            elif images.shape[2] == 3:
                # [B, num_cameras, C, H, W] -> [B, C, T, H, W]
                images = images.permute(0, 2, 1, 3, 4)
        elif images.dim() == 4:
            # Single image [B, C, H, W] -> [B, C, 1, H, W]
            images = images.unsqueeze(2)

        # Encode with VAE (float32 for stability)
        images_for_vae = images.to(dtype=torch.float32)

        # VAE encoding - gradients flow only if VAE is not frozen
        if self.freeze_vae or self.freeze_backbone:
            with torch.no_grad():
                latents = self.vae.encode(images_for_vae).latent_dist.sample()
        else:
            latents = self.vae.encode(images_for_vae).latent_dist.sample()

        # Wan2.2 VAE uses channel-wise normalization instead of simple scaling
        # latents shape: [B, C, T, H, W] where C = z_dim (48)
        if hasattr(self.vae.config, 'latents_mean') and self.vae.config.latents_mean is not None:
            latents_mean = torch.tensor(self.vae.config.latents_mean, device=device, dtype=latents.dtype)
            latents_std = torch.tensor(self.vae.config.latents_std, device=device, dtype=latents.dtype)
            # Reshape for broadcasting: [C] -> [1, C, 1, 1, 1]
            latents_mean = latents_mean.view(1, -1, 1, 1, 1)
            latents_std = latents_std.view(1, -1, 1, 1, 1)
            latents = (latents - latents_mean) / latents_std
        elif hasattr(self.vae.config, 'scaling_factor') and self.vae.config.scaling_factor is not None:
            latents = latents * self.vae.config.scaling_factor

        # Convert latents to transformer dtype
        latents = latents.to(dtype=self.dtype, device=device)

        # Default timestep = 0 (clean image, no noise)
        if timestep is None:
            timestep = torch.zeros(batch_size, device=device, dtype=torch.long)

        # Clear previous outputs
        self._layer_outputs.clear()

        # Forward through transformer (hooks capture intermediate states)
        # Need to provide encoder_hidden_states (text conditioning) - use dummy
        encoder_hidden_states = torch.zeros(
            batch_size, 1, self.transformer.config.text_dim,
            device=device, dtype=self.dtype
        )

        # Transformer forward - gradients flow if not frozen
        if self.freeze_backbone:
            with torch.no_grad():
                _ = self.transformer(
                    hidden_states=latents,
                    timestep=timestep,
                    encoder_hidden_states=encoder_hidden_states,
                )
        else:
            _ = self.transformer(
                hidden_states=latents,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
            )

        # Collect and process hooked outputs
        layer_hidden_states = []
        for expert_idx, wan_idx in enumerate(self.layer_indices):
            if wan_idx not in self._layer_outputs:
                raise RuntimeError(f"Hook for layer {wan_idx} did not capture output")

            hidden = self._layer_outputs[wan_idx]  # [B, num_tokens, wan2_hidden_dim]

            # Detach if frozen (hooks capture tensors in computation graph)
            if self.freeze_backbone:
                hidden = hidden.detach()

            # Ensure hidden is the right shape
            if hidden.dim() == 4:
                # [B, T, num_tokens, dim] -> [B, T*num_tokens, dim]
                hidden = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1])

            # Resize token compressor if needed (first forward pass)
            num_tokens = hidden.shape[1]
            self._ensure_token_compressor(num_tokens, device)

            # Token compression: [B, num_tokens, dim] -> [B, vam_num_tokens, dim]
            # Cast to compressor dtype for computation
            hidden = hidden.to(dtype=self.token_compressor.weight.dtype)
            hidden = hidden.transpose(1, 2)  # [B, dim, num_tokens]
            hidden = self.token_compressor(hidden)  # [B, dim, vam_num_tokens]
            hidden = hidden.transpose(1, 2)  # [B, vam_num_tokens, dim]

            # Project to expert dimension: [B, vam_num_tokens, wan2_dim] -> [B, vam_num_tokens, vam_hidden_dim]
            hidden = hidden.to(dtype=self.projections[expert_idx].weight.dtype)
            hidden = self.projections[expert_idx](hidden)

            layer_hidden_states.append(hidden)

        return layer_hidden_states

    def get_num_params(self) -> dict[str, int]:
        """Return parameter counts for logging."""
        vae_params = sum(p.numel() for p in self.vae.parameters())
        transformer_params = sum(p.numel() for p in self.transformer.parameters())
        projection_params = sum(p.numel() for p in self.projections.parameters())
        compressor_params = sum(p.numel() for p in self.token_compressor.parameters())

        return {
            "vae": vae_params,
            "transformer": transformer_params,
            "projections": projection_params,
            "token_compressor": compressor_params,
            "total": vae_params + transformer_params + projection_params + compressor_params,
        }

    def get_trainable_params(self) -> dict[str, int]:
        """Return trainable parameter counts."""
        vae_params = sum(p.numel() for p in self.vae.parameters() if p.requires_grad)
        transformer_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        projection_params = sum(p.numel() for p in self.projections.parameters() if p.requires_grad)
        compressor_params = sum(p.numel() for p in self.token_compressor.parameters() if p.requires_grad)

        return {
            "vae": vae_params,
            "transformer": transformer_params,
            "projections": projection_params,
            "token_compressor": compressor_params,
            "total": vae_params + transformer_params + projection_params + compressor_params,
        }

    def __repr__(self) -> str:
        return (
            f"Wan22VAM(\n"
            f"  model_id={self.model_id},\n"
            f"  wan2_hidden_dim={self.wan2_hidden_dim},\n"
            f"  wan2_num_layers={self.wan2_num_layers},\n"
            f"  num_expert_layers={self.num_expert_layers},\n"
            f"  vam_hidden_dim={self.vam_hidden_dim},\n"
            f"  vam_num_tokens={self.vam_num_tokens},\n"
            f"  freeze_backbone={self.freeze_backbone},\n"
            f")"
        )
