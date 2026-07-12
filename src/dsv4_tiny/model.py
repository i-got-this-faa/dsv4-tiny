"""DSV4TinyForCausalLM — HuggingFace-compatible causal LM with DSv4 tiered KV cache.

Architecture:
    - Embedding → 24 decoder layers (SWA/CSA/HCA) → final norm → LM head
    - Each decoder layer: input_layernorm → DSV4Attention → residual → post_attn_layernorm → MLP → residual
    - DSV4Cache attached for three-tier KV management
    - MTP auxiliary head (kept from base model)

Weight loading:
    - Loads Qwen3.5-0.8B via HuggingFace
    - Replaces all attention layers with DSV4Attention
    - Copies Q/K/V/O projection weights where available
    - Randomly initializes compression weights
"""


from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DSV4TinyConfig
from .attention import DSV4Attention
from .cache import DSV4Cache
from .utils import rms_norm
from torch.utils.checkpoint import checkpoint as grad_checkpoint



class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.weight, self.variance_epsilon)


class MLP(nn.Module):
    """SwiGLU MLP (matching Qwen3.5 architecture)."""

    def __init__(self, config: DSV4TinyConfig):
        super().__init__()
        d = config.hidden_size
        d_ff = config.intermediate_size

        self.gate_proj = nn.Linear(d, d_ff, bias=False)
        self.up_proj = nn.Linear(d, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class DecoderLayer(nn.Module):
    """Single decoder layer: norm → attention → residual → norm → MLP → residual."""

    def __init__(self, config: DSV4TinyConfig, layer_idx: int, shared_W_DQ: Optional[nn.Linear] = None):
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = DSV4Attention(config, layer_idx, shared_W_DQ=shared_W_DQ)
        self.mlp = MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: Optional[DSV4Cache] = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(hidden_states, cache=cache)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class DSV4TinyForCausalLM(nn.Module):
    """DSV4-Tiny model with three-tier heterogeneous KV cache.

    HuggingFace-compatible for training and inference.
    """

    def __init__(self, config: DSV4TinyConfig):
        super().__init__()
        self.config = config

        # ── Embedding ──
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # ── Shared decomposed query projection ──
        if config.shared_W_DQ:
            c_I = config.indexer_dim
            n_Ih = config.indexer_heads
            self.W_DQ = nn.Linear(config.hidden_size, c_I * n_Ih, bias=False)
        else:
            self.W_DQ = None

        # ── Decoder layers ──
        self.layers = nn.ModuleList([
            DecoderLayer(config, i, shared_W_DQ=self.W_DQ) for i in range(config.num_hidden_layers)
        ])

        # ── Final norm ──
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # ── LM head ──
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # ── Tie embeddings ──
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        # ── KV Cache ──
        self.cache = DSV4Cache(config, shared_W_DQ=self.W_DQ)

        # ── Initialize weights ──
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize all model weights."""
        # Embedding
        nn.init.normal_(self.embed_tokens.weight, mean=0.0, std=0.02)
        # LM head (may be tied to embedding)
        if not self.config.tie_word_embeddings:
            nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)

    @classmethod
    def from_pretrained(
        cls,
        base_model_path: str = "Qwen/Qwen3.5-0.8B",
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.bfloat16,
    ) -> "DSV4TinyForCausalLM":
        """Load Qwen3.5-0.8B weights and build DSV4-Tiny model.

        Step 1: Create DSV4TinyConfig from HF config
        Step 2: Create the DSV4TinyForCausalLM skeleton
        Step 3: Load and map weights from the pretrained model

        Args:
            base_model_path: HuggingFace model ID or local path.
            device: Target device.
            dtype: Model dtype.

        Returns:
            DSV4TinyForCausalLM with weights loaded.
        """
        from transformers import AutoModelForCausalLM

        cfg = DSV4TinyConfig.from_base_model()
        model = cls(cfg)

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(dtype=dtype)

        # Load pretrained model
        print(f"Loading base model from {base_model_path}...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # ── Copy embeddings ──
        state = base.state_dict()
        model.embed_tokens.weight.data.copy_(state["model.embed_tokens.weight"].to(dtype=dtype))

        # ── Copy layer weights ──
        for i in range(cfg.num_hidden_layers):
            layer = model.layers[i]
            prefix = f"model.layers.{i}"

            # Copy RMS norm weights
            layer.input_layernorm.weight.data.copy_(
                state[f"{prefix}.input_layernorm.weight"].to(dtype=dtype)
            )
            layer.post_attention_layernorm.weight.data.copy_(
                state[f"{prefix}.post_attention_layernorm.weight"].to(dtype=dtype)
            )

            # Copy attention Q projection
            attn = layer.self_attn

            # Q projection — always present in both full and linear attention
            q_key = f"{prefix}.self_attn.q_proj.weight"
            if q_key in state:
                attn.q_proj.weight.data.copy_(state[q_key].to(dtype=dtype))
            else:
                # Fallback: try linear_attn naming
                q_key = f"{prefix}.self_attn.q_a_proj.weight"
                if q_key in state:
                    # GatedDeltaNet has q_a + q_b projections
                    q_a = state[q_key].to(dtype=dtype)
                    q_b = state[f"{prefix}.self_attn.q_b_proj.weight"].to(dtype=dtype)
                    # Combine: q_proj = q_b @ q_a
                    combined = q_b @ q_a
                    attn.q_proj.weight.data.copy_(combined)
                else:
                    print(f"  Warning: no Q weight for layer {i}, using random init")

            # K/V projections (only for SWA layers — copy if available)
            k_key = f"{prefix}.self_attn.k_proj.weight"
            v_key = f"{prefix}.self_attn.v_proj.weight"

            # K/V projections (every layer now has k_proj/v_proj)
            if k_key in state:
                attn.k_proj.weight.data.copy_(state[k_key].to(dtype=dtype))
            if v_key in state:
                attn.v_proj.weight.data.copy_(state[v_key].to(dtype=dtype))

            # O projection
            o_key = f"{prefix}.self_attn.o_proj.weight"
            if o_key in state:
                attn.o_proj.weight.data.copy_(state[o_key].to(dtype=dtype))

            # Copy MLP weights
            layer.mlp.gate_proj.weight.data.copy_(
                state[f"{prefix}.mlp.gate_proj.weight"].to(dtype=dtype)
            )
            layer.mlp.up_proj.weight.data.copy_(
                state[f"{prefix}.mlp.up_proj.weight"].to(dtype=dtype)
            )
            layer.mlp.down_proj.weight.data.copy_(
                state[f"{prefix}.mlp.down_proj.weight"].to(dtype=dtype)
            )

        # ── Copy final norm ──
        model.norm.weight.data.copy_(state["model.norm.weight"].to(dtype=dtype))

        # ── Copy LM head (may be tied) ──
        if "lm_head.weight" in state:
            model.lm_head.weight.data.copy_(state["lm_head.weight"].to(dtype=dtype))

        # Move to device
        model = model.to(device=device)

        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        return model

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        return_dict: bool = True,
        cpu_offload_lm_head: bool = False,
        use_grad_checkpoint: bool = False,
    ) -> dict:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len) token indices.
            attention_mask: (batch_size, seq_len) — unused in this v1 implementation
                (causal mask is used).
            labels: (batch_size, seq_len) target token indices for loss computation.
            use_cache: Whether to use the DSV4 cache.
            return_dict: Whether to return a dict (vs tuple).
            cpu_offload_lm_head: Move lm_head to CPU for logits computation (saves GPU memory
                when lm_head weight doesn't need gradients, e.g. frozen training).
            use_grad_checkpoint: Use gradient checkpointing per decoder layer to trade
                compute for memory (reduces activation memory ~2-3x).

        Returns:
            Dict with keys: 'logits', 'loss' (if labels provided), 'cache_stats'.
        """
        b, seq_len = input_ids.shape
        device = input_ids.device
        dtype = next(self.parameters()).dtype

        # Reset cache for new sequence
        if use_cache:
            self.cache.reset()

        # Embed
        hidden_states = self.embed_tokens(input_ids)  # (b, seq_len, d)

        # Run through decoder layers with optional gradient checkpointing
        for layer in self.layers:
            if use_grad_checkpoint and self.training:
                def layer_fn(h, l=layer, c=self.cache, uc=use_cache):
                    kwargs = {"cache": c} if uc else {}
                    return l(h, **kwargs)
                hidden_states = grad_checkpoint(layer_fn, hidden_states, use_reentrant=False)
            else:
                kwargs = {"cache": self.cache} if use_cache else {}
                hidden_states = layer(hidden_states, **kwargs)

        # Final norm
        hidden_states = self.norm(hidden_states)

        # LM head — optionally on CPU to save GPU memory
        if cpu_offload_lm_head and self.lm_head.weight.device.type == "cuda":
            # Offload lm_head computation to CPU while preserving autograd.
            # Since lm_head.weight is tied to embed_tokens.weight, moving the
            # module to CPU temporarily also moves the embedding — both come
            # back when we restore. This saves ~vocab_size*seq_len bytes of
            # GPU memory for the logits tensor.
            orig_device = next(self.parameters()).device
            orig_dtype = self.lm_head.weight.dtype
            hidden_states_cpu = hidden_states.to("cpu")  # preserves grad
            self.lm_head = self.lm_head.to("cpu")
            logits = self.lm_head(hidden_states_cpu.to(orig_dtype))
            self.lm_head = self.lm_head.to(orig_device)
            logits = logits.to(device, dtype=dtype)
        else:
            logits = self.lm_head(hidden_states)  # (b, seq_len, vocab_size)

        # Loss
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        if return_dict:
            result = {"logits": logits}
            if loss is not None:
                result["loss"] = loss
            if use_cache:
                result["cache_stats"] = self.cache.cache_stats
                recon_loss = self.cache.reconstruction_loss
                if recon_loss is not None:
                    result["recon_loss"] = recon_loss
            return result

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            input_ids: (batch_size, seq_len) prompt tokens.
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k filtering.
            top_p: Top-p (nucleus) filtering.
            eos_token_id: End-of-sequence token ID.

        Returns:
            (batch_size, prompt_len + generated_len) tokens.
        """
        device = next(self.parameters()).device
        if eos_token_id is None:
            eos_token_id = 248044  # Qwen3.5 EOS

        self.eval()
        self.cache.reset()

        batch_size = input_ids.shape[0]
        generated = input_ids.clone()

        # Prefill: process the full prompt
        with torch.no_grad():
            # Run full prompt through the model to populate cache
            _ = self.forward(input_ids, use_cache=True)

        # Generate autoregressively
        for _ in range(max_new_tokens):
            # Get the last token
            last_token = generated[:, -1:]  # (b, 1)

            # Forward pass with just the last token (cache has all previous)
            outputs = self.forward(last_token, use_cache=True)
            next_logits = outputs["logits"][:, -1, :]  # (b, vocab_size)

            # Temperature
            if temperature > 0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k > 0:
                top_k_values, _ = torch.topk(next_logits, top_k, dim=-1)
                min_top_k = top_k_values[:, -1].unsqueeze(-1)
                next_logits[next_logits < min_top_k] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above top_p
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_logits[indices_to_remove] = float("-inf")

            # Sample or greedy
            if temperature == 0:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=-1)

            # Check for EOS
            if (next_token == eos_token_id).any():
                break

        self.train()
        return generated
