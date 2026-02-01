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

import copy

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


def get_intermediate_size(hidden_dim, ffn_dim_multiplier=4, multiple_of=256):
    hidden_dim = int(2 * hidden_dim / 3)
    hidden_dim = int(ffn_dim_multiplier * hidden_dim)
    hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
    return hidden_dim


class TransformerVAM(nn.Module):
    """
    Transformer-based VAM (Video Action Model) that interacts with the action expert
    layer-by-layer, similar to how the VLM does.

    Architecture:
    - Input projection to match hidden dimension
    - Learnable tokens that get processed through transformer layers
    - Each layer produces K/V states for the corresponding expert layer to cross-attend to

    TODO: Replace the input processing with your actual video encoder.
    """

    def __init__(
        self,
        input_dim: int,  # Input feature dimension (e.g., from video encoder)
        hidden_dim: int,  # Hidden dimension (should match expert's expected K/V input)
        num_layers: int,  # Number of transformer layers (should match expert layers)
        num_tokens: int = 64,  # Number of VAM tokens
        num_heads: int = 8,  # Number of attention heads
        head_dim: int = 64,  # Dimension per head
        intermediate_size: int = None,  # FFN intermediate size
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim

        if intermediate_size is None:
            intermediate_size = get_intermediate_size(hidden_dim)

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Learnable VAM tokens
        self.vam_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden_dim) * 0.02)

        # Transformer layers for VAM
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                VAMTransformerLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    head_dim=head_dim,
                    intermediate_size=intermediate_size,
                    dropout=dropout,
                )
            )

        # Final layer norm
        self.norm = nn.RMSNorm(hidden_dim)

        # K/V projection for cross-attention with expert
        # These project VAM hidden states to K/V for the expert
        self.num_kv_heads = num_heads  # Can be different for GQA
        self.kv_dim = self.num_kv_heads * head_dim
        self.k_proj = nn.Linear(hidden_dim, self.kv_dim)
        self.v_proj = nn.Linear(hidden_dim, self.kv_dim)

    def forward(
        self,
        input_features: torch.Tensor = None,  # [B, seq_len, input_dim] - e.g., video features
        batch_size: int = None,
        device: torch.device = None,
    ) -> list[torch.Tensor]:
        """
        Process input through VAM transformer layers.

        Args:
            input_features: Optional input features to condition on [B, seq_len, input_dim]
            batch_size: Batch size (used if input_features is None)
            device: Device to create tensors on

        Returns:
            List of hidden states, one per layer: [num_layers x [B, num_tokens, hidden_dim]]
        """
        if input_features is not None:
            batch_size = input_features.shape[0]
            device = input_features.device
            dtype = input_features.dtype

            # Project input features
            projected = self.input_proj(input_features)  # [B, seq_len, hidden_dim]
            pooled = projected.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]

            # Initialize VAM tokens with input conditioning
            hidden_states = self.vam_tokens.expand(batch_size, -1, -1).to(device=device, dtype=dtype)
            hidden_states = hidden_states + pooled  # [B, num_tokens, hidden_dim]
        else:
            assert batch_size is not None and device is not None
            hidden_states = self.vam_tokens.expand(batch_size, -1, -1).to(device)

        # Process through transformer layers, collecting per-layer outputs
        layer_outputs = []
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            layer_outputs.append(hidden_states)

        # Apply final norm to last layer output
        layer_outputs[-1] = self.norm(layer_outputs[-1])

        return layer_outputs

    def get_kv_states(
        self,
        hidden_states: torch.Tensor,  # [B, num_tokens, hidden_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Project hidden states to K/V for cross-attention.

        Args:
            hidden_states: [B, num_tokens, hidden_dim]

        Returns:
            key_states: [B, num_tokens, num_kv_heads, head_dim]
            value_states: [B, num_tokens, num_kv_heads, head_dim]
        """
        batch_size, seq_len = hidden_states.shape[:2]

        key_states = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        value_states = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)

        return key_states, value_states


class VAMTransformerLayer(nn.Module):
    """Single transformer layer for VAM with self-attention and FFN."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        head_dim: int,
        intermediate_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim

        # Self-attention
        self.input_layernorm = nn.RMSNorm(hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.k_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.v_proj = nn.Linear(hidden_dim, num_heads * head_dim)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_dim)

        # FFN
        self.post_attention_layernorm = nn.RMSNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, intermediate_size)
        self.up_proj = nn.Linear(hidden_dim, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, seq_len, hidden_dim]

        Returns:
            output: [B, seq_len, hidden_dim]
        """
        batch_size, seq_len = hidden_states.shape[:2]
        residual = hidden_states

        # Self-attention
        hidden_states = self.input_layernorm(hidden_states)

        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)

        # Transpose for attention: [B, num_heads, seq_len, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: [B, seq_len, num_heads * head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)
        attn_output = self.o_proj(attn_output)

        hidden_states = residual + attn_output

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)

        # SwiGLU-style FFN
        gate = torch.nn.functional.silu(self.gate_proj(hidden_states))
        up = self.up_proj(hidden_states)
        hidden_states = gate * up
        hidden_states = self.down_proj(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class GatedFusion(nn.Module):
    """
    Learnable gating mechanism to fuse VLM and VAM cross-attention outputs.

    Supports multiple fusion modes:
    - "sigmoid_gate": Learn a gate g, output = g * vlm_out + (1-g) * vam_out
    - "softmax_gate": Softmax over [vlm, vam] attention weights
    - "mlp_fusion": Concatenate and project through MLP
    - "film": FiLM-style modulation where VAM modulates VLM output
    """

    def __init__(
        self,
        hidden_size: int,
        fusion_mode: str = "sigmoid_gate",
        num_heads: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.fusion_mode = fusion_mode
        self.num_heads = num_heads

        if fusion_mode == "sigmoid_gate":
            # Query-dependent gating: gate = sigmoid(W @ query + b)
            self.gate_proj = nn.Linear(hidden_size, hidden_size)
            self.gate_bias = nn.Parameter(torch.zeros(hidden_size))

        elif fusion_mode == "softmax_gate":
            # Learn attention weights over [vlm, vam] for each position
            self.gate_query = nn.Linear(hidden_size, hidden_size)
            self.gate_keys = nn.Parameter(torch.randn(2, hidden_size) * 0.02)  # 2 = [vlm, vam]

        elif fusion_mode == "mlp_fusion":
            # Concatenate vlm_out and vam_out, project through MLP
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Dropout(dropout),
            )

        elif fusion_mode == "film":
            # FiLM: VAM produces scale (gamma) and shift (beta) for VLM output
            self.film_generator = nn.Linear(hidden_size, hidden_size * 2)  # outputs [gamma, beta]

        else:
            raise ValueError(f"Unknown fusion_mode: {fusion_mode}")

    def forward(
        self,
        vlm_attn_out: torch.Tensor,  # [B, seq_len, hidden_size]
        vam_attn_out: torch.Tensor,  # [B, seq_len, hidden_size]
        query: torch.Tensor = None,   # [B, seq_len, hidden_size] - original query for gating
    ) -> torch.Tensor:
        """
        Fuse VLM and VAM attention outputs.

        Args:
            vlm_attn_out: Output from VLM cross-attention [B, seq_len, hidden_size]
            vam_attn_out: Output from VAM cross-attention [B, seq_len, hidden_size]
            query: Original query embeddings for query-dependent gating [B, seq_len, hidden_size]

        Returns:
            Fused output [B, seq_len, hidden_size]
        """
        # Ensure consistent dtype
        compute_dtype = vlm_attn_out.dtype

        if self.fusion_mode == "sigmoid_gate":
            # Query-dependent sigmoid gate
            if query is None:
                query = vlm_attn_out  # fallback to using vlm_out as query
            query = query.to(dtype=self.gate_proj.weight.dtype)
            gate = torch.sigmoid(self.gate_proj(query) + self.gate_bias)
            gate = gate.to(dtype=compute_dtype)
            fused = gate * vlm_attn_out + (1 - gate) * vam_attn_out

        elif self.fusion_mode == "softmax_gate":
            # Softmax attention over [vlm, vam]
            if query is None:
                query = vlm_attn_out
            query = query.to(dtype=self.gate_query.weight.dtype)
            query_proj = self.gate_query(query)  # [B, seq_len, hidden_size]
            # Compute attention scores with learned keys
            scores = torch.einsum('bsh,kh->bsk', query_proj, self.gate_keys)  # [B, seq_len, 2]
            weights = torch.softmax(scores, dim=-1).to(dtype=compute_dtype)  # [B, seq_len, 2]
            # Stack and weight
            stacked = torch.stack([vlm_attn_out, vam_attn_out], dim=-2)  # [B, seq_len, 2, hidden_size]
            fused = torch.einsum('bsk,bskh->bsh', weights, stacked)  # [B, seq_len, hidden_size]

        elif self.fusion_mode == "mlp_fusion":
            # Concatenate and project
            concat = torch.cat([vlm_attn_out, vam_attn_out], dim=-1)  # [B, seq_len, hidden_size*2]
            concat = concat.to(dtype=self.fusion_mlp[0].weight.dtype)
            fused = self.fusion_mlp(concat).to(dtype=compute_dtype)  # [B, seq_len, hidden_size]

        elif self.fusion_mode == "film":
            # FiLM modulation: VAM modulates VLM
            vam_for_film = vam_attn_out.to(dtype=self.film_generator.weight.dtype)
            film_params = self.film_generator(vam_for_film).to(dtype=compute_dtype)  # [B, seq_len, hidden_size*2]
            gamma, beta = film_params.chunk(2, dim=-1)  # each [B, seq_len, hidden_size]
            fused = (1 + gamma) * vlm_attn_out + beta

        return fused


class SmolVLMWithExpertModel(nn.Module):
    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        load_vlm_weights: bool = True,
        train_expert_only: bool = True,
        freeze_vision_encoder: bool = False,
        attention_mode: str = "self_attn",
        num_expert_layers: int = -1,
        num_vlm_layers: int = -1,
        self_attn_every_n_layers: int = -1,
        expert_width_multiplier: float = 0.5,
        device: str = "auto",
        # VAM (Video Action Model) cross-attention parameters
        vam_hidden_dim: int = None,  # Set to enable VAM cross-attention (e.g., 512, 1024)
        vam_fusion_mode: str = "sigmoid_gate",  # "sigmoid_gate", "softmax_gate", "mlp_fusion", "film"
        vam_num_tokens: int = 64,  # Number of VAM output tokens
        use_placeholder_vam: bool = True,  # Use placeholder VAM for testing (set False when you have real VAM)
        # Wan2.2-TI2V-5B VAM configuration
        use_wan2_vam: bool = False,  # Use Wan2.2-TI2V-5B as VAM (5B params, 30 layers)
        wan2_model_id: str = "Wan-AI/Wan2.2-TI2V-5B-Diffusers",  # HuggingFace model ID
        wan2_freeze: bool = False,  # Freeze entire Wan2.2 backbone (False = fine-tune)
        wan2_freeze_vae: bool = True,  # Freeze VAE (recommended)
        wan2_freeze_text_embedder: bool = True,  # Freeze text embedder (recommended)
        wan2_trainable_layers: int = -1,  # Train last N layers (-1 = all)
    ):
        super().__init__()
        if load_vlm_weights:
            print(f"Loading  {model_id} weights ...")
            self.vlm = AutoModelForImageTextToText.from_pretrained(
                model_id,
                device_map=device,
                torch_dtype="bfloat16",
                low_cpu_mem_usage=True,
            )
            config = self.vlm.config
        else:
            config = AutoConfig.from_pretrained(model_id)
            self.vlm = SmolVLMForConditionalGeneration(config=config)
        self.processor = AutoProcessor.from_pretrained(model_id)
        if num_vlm_layers > 0:
            print(f"Reducing the number of VLM layers to {num_vlm_layers} ...")
            self.get_vlm_model().text_model.layers = self.get_vlm_model().text_model.layers[:num_vlm_layers]
        self.num_vlm_layers = len(self.get_vlm_model().text_model.layers)
        self.config = config
        # Smaller lm expert
        lm_expert_config = copy.deepcopy(config.text_config)
        hidden_size = lm_expert_config.hidden_size
        lm_expert_config.hidden_size = int(hidden_size * expert_width_multiplier)  # hidden_size // 2
        lm_expert_config.intermediate_size = get_intermediate_size(int(hidden_size * expert_width_multiplier))
        lm_expert_config.num_hidden_layers = self.num_vlm_layers
        if num_expert_layers > 0:
            assert len(self.get_vlm_model().text_model.layers) % num_expert_layers == 0, (
                f"Number of layers in the VLM {len(self.get_vlm_model().text_model.layers)} are not multiple of num_expert_layers {num_expert_layers}"
            )
            lm_expert_config.num_hidden_layers = num_expert_layers
        self.lm_expert = AutoModel.from_config(lm_expert_config)

        self.num_expert_layers = len(self.lm_expert.layers)
        self.self_attn_every_n_layers = self_attn_every_n_layers
        if "cross" in attention_mode:
            # Reshape qkv projections to have the same input dimension as the vlm
            for layer_idx in range(len(self.lm_expert.layers)):
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    continue
                self.lm_expert.layers[layer_idx].self_attn.k_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
                self.lm_expert.layers[layer_idx].self_attn.v_proj = nn.Linear(
                    config.text_config.num_key_value_heads * config.text_config.head_dim,
                    lm_expert_config.num_key_value_heads * lm_expert_config.head_dim,
                    bias=lm_expert_config.attention_bias,
                )
        # Remove unused embed_tokens
        self.lm_expert.embed_tokens = None

        self.num_attention_heads = self.config.text_config.num_attention_heads
        self.num_key_value_heads = self.config.text_config.num_key_value_heads

        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_mode = attention_mode
        self.expert_hidden_size = lm_expert_config.hidden_size

        # ==================== VAM Cross-Attention Setup ====================
        self.vam_enabled = vam_hidden_dim is not None
        self.vam_hidden_dim = vam_hidden_dim
        self.vam_fusion_mode = vam_fusion_mode

        if self.vam_enabled:
            print(f"Initializing VAM cross-attention with hidden_dim={vam_hidden_dim}, fusion_mode={vam_fusion_mode}")

            # Expert dimensions for K/V projections
            expert_kv_dim = lm_expert_config.num_key_value_heads * lm_expert_config.head_dim

            # VAM K/V projection layers (one per expert layer)
            # These project VAM hidden states to expert's K/V space
            self.vam_k_projs = nn.ModuleList()
            self.vam_v_projs = nn.ModuleList()
            self.vam_fusions = nn.ModuleList()

            for layer_idx in range(len(self.lm_expert.layers)):
                # Check if this layer uses cross-attention (not self-attention)
                if self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0:
                    # Self-attention layer - no VAM cross-attention
                    self.vam_k_projs.append(None)
                    self.vam_v_projs.append(None)
                    self.vam_fusions.append(None)
                else:
                    # Cross-attention layer - add VAM projections
                    self.vam_k_projs.append(
                        nn.Linear(vam_hidden_dim, expert_kv_dim, bias=lm_expert_config.attention_bias)
                    )
                    self.vam_v_projs.append(
                        nn.Linear(vam_hidden_dim, expert_kv_dim, bias=lm_expert_config.attention_bias)
                    )
                    # Gated fusion to combine VLM and VAM attention outputs
                    # Note: fusion operates on attention output dimension (num_heads * head_dim)
                    # which is BEFORE o_proj reduces it back to hidden_size
                    attn_output_dim = lm_expert_config.num_attention_heads * lm_expert_config.head_dim
                    self.vam_fusions.append(
                        GatedFusion(
                            hidden_size=attn_output_dim,
                            fusion_mode=vam_fusion_mode,
                            num_heads=lm_expert_config.num_attention_heads,
                        )
                    )

            num_cross_attn_layers = sum(1 for p in self.vam_k_projs if p is not None)
            print(f"  Created {num_cross_attn_layers} VAM cross-attention layers (expert has {self.num_expert_layers} layers)")

            # Initialize VAM for layer-by-layer interaction
            self.use_placeholder_vam = use_placeholder_vam
            self.use_wan2_vam = use_wan2_vam

            if use_wan2_vam:
                # Use Wan2.2-TI2V-5B as VAM (5B params, 30 layers, 3072 hidden dim)
                print(f"  Initializing Wan2.2-TI2V-5B VAM...")
                from lerobot.policies.smolvla.wan_vam import Wan22VAM
                self.vam = Wan22VAM(
                    model_id=wan2_model_id,
                    num_expert_layers=self.num_expert_layers,
                    vam_hidden_dim=vam_hidden_dim,
                    vam_num_tokens=vam_num_tokens,
                    freeze_backbone=wan2_freeze,
                    freeze_vae=wan2_freeze_vae,
                    freeze_text_embedder=wan2_freeze_text_embedder,
                    trainable_layers=wan2_trainable_layers,
                )
                print(f"  Wan2.2 VAM initialized: {vam_num_tokens} tokens, {self.num_expert_layers} expert layers")
            elif use_placeholder_vam:
                print(f"  Using TransformerVAM with {vam_num_tokens} tokens and {self.num_expert_layers} layers (matching expert)")
                self.vam = TransformerVAM(
                    input_dim=config.text_config.hidden_size,  # VLM hidden size as input
                    hidden_dim=vam_hidden_dim,
                    num_layers=self.num_expert_layers,  # Match expert layer count
                    num_tokens=vam_num_tokens,
                    num_heads=8,  # Can be configured
                    head_dim=64,  # Can be configured
                )
            else:
                self.vam = None
                print("  VAM disabled - you must provide vam_layer_hidden_states to forward()")
        else:
            self.use_placeholder_vam = False
            self.use_wan2_vam = False
            self.vam = None
        # ====================================================================

        self.set_requires_grad()

    def get_vlm_model(self):
        return self.vlm.model

    def set_requires_grad(self):
        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()
            for params in self.get_vlm_model().vision_model.parameters():
                params.requires_grad = False
        if self.train_expert_only:
            self.vlm.eval()
            for params in self.vlm.parameters():
                params.requires_grad = False
        else:
            # To avoid unused params issue with distributed training
            last_layers = [self.num_vlm_layers - 1]
            if (
                self.num_vlm_layers != self.num_expert_layers
                and self.num_vlm_layers % self.num_expert_layers == 0
            ):
                last_layers.append(self.num_vlm_layers - 2)
            frozen_layers = [
                "lm_head",
                "text_model.model.norm.weight",
            ]
            for layer in last_layers:
                frozen_layers.append(f"text_model.model.layers.{layer}.")

            for name, params in self.vlm.named_parameters():
                if any(k in name for k in frozen_layers):
                    params.requires_grad = False
        # To avoid unused params issue with distributed training
        for name, params in self.lm_expert.named_parameters():
            if "lm_head" in name:
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.freeze_vision_encoder:
            self.get_vlm_model().vision_model.eval()

        if self.train_expert_only:
            self.vlm.eval()

    def embed_image(self, image: torch.Tensor):
        patch_attention_mask = None
        # Get sequence from the vision encoder
        image_hidden_states = (
            self.get_vlm_model()
            .vision_model(
                pixel_values=image.to(dtype=self.get_vlm_model().vision_model.dtype),
                patch_attention_mask=patch_attention_mask,
            )
            .last_hidden_state
        )
        # Modality projection & resampling
        image_hidden_states = self.get_vlm_model().connector(image_hidden_states)
        return image_hidden_states

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.get_vlm_model().text_model.get_input_embeddings()(tokens)

    def forward_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
    ) -> list[torch.Tensor]:
        query_states = []
        key_states = []
        value_states = []
        for i, hidden_states in enumerate(inputs_embeds):
            layer = model_layers[i][layer_idx]
            if hidden_states is None or layer is None:
                continue
            hidden_states = layer.input_layernorm(hidden_states)

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            query_states.append(query_state)
            key_states.append(key_state)
            value_states.append(value_state)

        # B,L,H,D with L sequence length, H number of heads, D head dim
        # concatenate on the number of embeddings/tokens
        query_states = torch.cat(query_states, dim=1)
        key_states = torch.cat(key_states, dim=1)
        value_states = torch.cat(value_states, dim=1)
        seq_len = query_states.shape[1]
        if seq_len < position_ids.shape[1]:
            _position_ids = position_ids[:, :seq_len]
            _attention_mask = attention_mask[:, :seq_len, :seq_len]
        else:
            _position_ids = position_ids
            _attention_mask = attention_mask

        attention_mask_ = _attention_mask
        position_ids_ = _position_ids

        query_states = apply_rope(query_states, position_ids_)
        key_states = apply_rope(key_states, position_ids_)

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                value_states = torch.cat([past_key_values[layer_idx]["value_states"], value_states], dim=1)

        attention_interface = self.get_attention_interface()

        att_output = attention_interface(
            attention_mask_, batch_size, head_dim, query_states, key_states, value_states
        )
        return [att_output], past_key_values

    def forward_cross_attn_layer(
        self,
        model_layers,
        inputs_embeds,
        layer_idx,
        position_ids,
        attention_mask,
        batch_size,
        head_dim,
        use_cache: bool = True,
        fill_kv_cache: bool = True,
        past_key_values=None,
        # VAM cross-attention inputs (layer-specific)
        vam_layer_hidden_state: torch.Tensor = None,  # [B, vam_seq_len, vam_hidden_dim] - hidden state for THIS layer
        vam_attention_mask: torch.Tensor = None,  # [B, action_seq, vam_seq_len]
    ) -> list[torch.Tensor]:
        attention_interface = self.get_attention_interface()

        att_outputs = []
        assert len(inputs_embeds) == 2 or (use_cache and past_key_values is not None and not fill_kv_cache), (
            f"Both len(inputs_embeds) == {len(inputs_embeds)} and past_key_values is {past_key_values}"
        )

        if len(inputs_embeds) == 2 and not past_key_values:
            # Prefix attention
            seq_len = inputs_embeds[0].shape[1]
            position_id, expert_position_id = position_ids[:, :seq_len], position_ids[:, seq_len:]
            prefix_attention_mask = attention_mask[:, :seq_len, :seq_len]

            layer = model_layers[0][layer_idx]

            hidden_states = layer.input_layernorm(inputs_embeds[0])

            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

            hidden_states = hidden_states.to(dtype=layer.self_attn.q_proj.weight.dtype)
            query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
            key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
            value_states = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            query_states = apply_rope(query_state, position_id)
            key_states = apply_rope(key_state, position_id)

            att_output = attention_interface(
                prefix_attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_outputs.append(att_output)
        else:
            expert_position_id = position_ids

        if use_cache and past_key_values is None:
            past_key_values = {}

        if use_cache:
            if fill_kv_cache:
                past_key_values[layer_idx] = {
                    "key_states": key_states,
                    "value_states": value_states,
                }
            else:
                # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                # the max len, then we (for instance) double the cache size. This implementation already exists
                # in `transformers`. (molbap)
                key_states = past_key_values[layer_idx]["key_states"]
                value_states = past_key_values[layer_idx]["value_states"]

        # Expert
        expert_layer = model_layers[1][layer_idx]
        if expert_layer is not None:
            expert_hidden_states = expert_layer.input_layernorm(inputs_embeds[1])

            expert_input_shape = expert_hidden_states.shape[:-1]
            expert_hidden_shape = (*expert_input_shape, -1, expert_layer.self_attn.head_dim)

            expert_hidden_states = expert_hidden_states.to(dtype=expert_layer.self_attn.q_proj.weight.dtype)
            expert_query_state = expert_layer.self_attn.q_proj(expert_hidden_states).view(expert_hidden_shape)

            _key_states = key_states.to(dtype=expert_layer.self_attn.k_proj.weight.dtype).view(
                *key_states.shape[:2], -1
            )
            expert_key_states = expert_layer.self_attn.k_proj(_key_states).view(
                *_key_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )  # k_proj should have same dim as kv

            _value_states = value_states.to(dtype=expert_layer.self_attn.v_proj.weight.dtype).view(
                *value_states.shape[:2], -1
            )
            expert_value_states = expert_layer.self_attn.v_proj(_value_states).view(
                *_value_states.shape[:-1], -1, expert_layer.self_attn.head_dim
            )

            expert_position_id = (
                expert_position_id - torch.min(expert_position_id, dim=1, keepdim=True).values
            )  # start from 0
            expert_attention_mask = attention_mask[
                :, -inputs_embeds[1].shape[1] :, : expert_key_states.shape[1] :
            ]  # take into account kv

            expert_query_states = apply_rope(expert_query_state, expert_position_id)

            # VLM cross-attention
            vlm_att_output = attention_interface(
                expert_attention_mask,
                batch_size,
                head_dim,
                expert_query_states,
                expert_key_states,
                expert_value_states,
            )

            # ==================== VAM Cross-Attention ====================
            if self.vam_enabled and vam_layer_hidden_state is not None and self.vam_k_projs[layer_idx] is not None:
                # Get VAM projection layers for this layer
                vam_k_proj = self.vam_k_projs[layer_idx]
                vam_v_proj = self.vam_v_projs[layer_idx]
                vam_fusion = self.vam_fusions[layer_idx]

                # Project VAM hidden states to K/V
                vam_hidden = vam_layer_hidden_state.to(dtype=vam_k_proj.weight.dtype)
                vam_key_states = vam_k_proj(vam_hidden).view(
                    *vam_hidden.shape[:-1], -1, expert_layer.self_attn.head_dim
                )  # [B, vam_seq, num_kv_heads, head_dim]
                vam_value_states = vam_v_proj(vam_hidden).view(
                    *vam_hidden.shape[:-1], -1, expert_layer.self_attn.head_dim
                )  # [B, vam_seq, num_kv_heads, head_dim]

                # Create VAM attention mask if not provided
                if vam_attention_mask is None:
                    # Default: all action tokens can attend to all VAM tokens
                    action_seq_len = expert_query_states.shape[1]
                    vam_seq_len = vam_key_states.shape[1]
                    vam_attention_mask = torch.ones(
                        batch_size, action_seq_len, vam_seq_len,
                        dtype=torch.bool, device=expert_query_states.device
                    )

                # VAM cross-attention (reuse same query from expert)
                vam_att_output = attention_interface(
                    vam_attention_mask,
                    batch_size,
                    head_dim,
                    expert_query_states,  # Same queries attend to VAM
                    vam_key_states,
                    vam_value_states,
                )

                # Fuse VLM and VAM attention outputs using learned gating
                att_output = vam_fusion(
                    vlm_attn_out=vlm_att_output,
                    vam_attn_out=vam_att_output,
                    query=vlm_att_output,  # Use VLM output as query for gating
                )
            else:
                # No VAM, use VLM attention output directly
                att_output = vlm_att_output
            # ==============================================================

            att_outputs.append(att_output)
        else:
            att_outputs.append(None)

        # att_output = att_output.to(dtype=models[i].dtype)
        return att_outputs, past_key_values

    def get_model_layers(self, models: list) -> list:
        vlm_layers = []
        expert_layers = []
        multiple_of = self.num_vlm_layers // self.num_expert_layers
        for i in range(self.num_vlm_layers):
            if multiple_of > 0 and i > 0 and i % multiple_of != 0:
                expert_layer = None
            else:
                expert_layer_index = i // multiple_of if multiple_of > 0 else i
                expert_layer = models[1].layers[expert_layer_index]
            vlm_layers.append(models[0].layers[i])
            expert_layers.append(expert_layer)
        return [vlm_layers, expert_layers]

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
        # VAM cross-attention inputs (layer-by-layer)
        vam_layer_hidden_states: list[torch.Tensor] | None = None,  # [num_expert_layers x [B, vam_seq, hidden_dim]]
        vam_attention_mask: torch.Tensor | None = None,  # [B, action_seq, vam_seq_len]
        # Raw images for Wan2.2 VAM (required when use_wan2_vam=True)
        raw_images: torch.Tensor | None = None,  # [B, num_cameras, C, H, W] for Wan2.2 VAM
    ):
        models = [self.get_vlm_model().text_model, self.lm_expert]
        model_layers = self.get_model_layers(models)
        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]
            device = hidden_states.device
            dtype = hidden_states.dtype

        # ==================== Generate VAM Layer Hidden States ====================
        # If VAM is enabled but no vam_layer_hidden_states provided, generate them
        if self.vam_enabled and vam_layer_hidden_states is None and self.vam is not None:
            if fill_kv_cache:
                # During KV cache filling, compute and cache VAM hidden states
                if self.use_wan2_vam:
                    # Wan2.2 VAM: process raw images through Wan2.2 model
                    if raw_images is not None:
                        vam_layer_hidden_states = self.vam(
                            images=raw_images,  # [B, num_cameras, C, H, W]
                        )  # Returns list of [B, num_tokens, hidden_dim], one per layer
                    else:
                        raise ValueError(
                            "Wan2.2 VAM requires raw_images input. "
                            "Pass raw_images to forward() when use_wan2_vam=True."
                        )
                elif self.use_placeholder_vam:
                    # TransformerVAM: process prefix embeddings
                    prefix_embeds = inputs_embeds[0]  # [B, prefix_seq, vlm_hidden]

                    if prefix_embeds is not None:
                        # Process through TransformerVAM to get per-layer outputs
                        vam_layer_hidden_states = self.vam(
                            input_features=prefix_embeds,  # Using prefix as proxy for input features
                        )  # Returns list of [B, num_tokens, hidden_dim], one per layer
                    else:
                        # Generate without conditioning
                        vam_layer_hidden_states = self.vam(
                            batch_size=batch_size,
                            device=device,
                        )

                # Cache VAM hidden states for inference
                if use_cache and past_key_values is None:
                    past_key_values = {}
                if use_cache:
                    past_key_values["vam_layer_hidden_states"] = vam_layer_hidden_states
            else:
                # During inference, retrieve cached VAM hidden states
                if use_cache and past_key_values is not None and "vam_layer_hidden_states" in past_key_values:
                    vam_layer_hidden_states = past_key_values["vam_layer_hidden_states"]
        # ==========================================================================

        # RMSNorm
        num_layers = self.num_vlm_layers
        head_dim = self.vlm.config.text_config.head_dim
        for layer_idx in range(num_layers):
            if (
                fill_kv_cache
                or "cross" not in self.attention_mode
                or (self.self_attn_every_n_layers > 0 and layer_idx % self.self_attn_every_n_layers == 0)
            ):
                att_outputs, past_key_values = self.forward_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                )
            else:
                # Get the appropriate VAM hidden state for this layer
                # Map VLM layer index to expert layer index
                multiple_of = self.num_vlm_layers // self.num_expert_layers
                expert_layer_idx = layer_idx // multiple_of if multiple_of > 0 else layer_idx

                vam_layer_hidden_state = None
                if vam_layer_hidden_states is not None and expert_layer_idx < len(vam_layer_hidden_states):
                    vam_layer_hidden_state = vam_layer_hidden_states[expert_layer_idx]

                att_outputs, past_key_values = self.forward_cross_attn_layer(
                    model_layers,
                    inputs_embeds,
                    layer_idx,
                    position_ids,
                    attention_mask,
                    batch_size,
                    head_dim,
                    use_cache=use_cache,
                    fill_kv_cache=fill_kv_cache,
                    past_key_values=past_key_values,
                    # VAM cross-attention (layer-specific)
                    vam_layer_hidden_state=vam_layer_hidden_state,
                    vam_attention_mask=vam_attention_mask,
                )
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = model_layers[i][layer_idx]
                att_output = (
                    att_outputs[i] if i < len(att_outputs) else att_outputs[0]
                )  # in case of self_attn
                if hidden_states is not None:
                    if layer is None:
                        outputs_embeds.append(hidden_states)
                        continue
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    att_out = att_output[:, start:end]
                    out_emb = layer.self_attn.o_proj(att_out)

                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end if len(att_outputs) == 1 else 0
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        attention_interface = self.eager_attention_forward
        return attention_interface

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.
        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5

        att_weights = att_weights.to(dtype=torch.float32)
        big_neg = torch.finfo(att_weights.dtype).min  # -2.3819763e38  # See gemma/modules.py
        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)
        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        return att_output
