"""Configuration for DSV4-Tiny: single source of truth for all architectural hyperparameters."""

from __future__ import annotations

import dataclasses
from typing import ClassVar, Literal


@dataclasses.dataclass(frozen=True)
class DSV4TinyConfig:
    """DSV4-Tiny hyperparameters derived from Qwen3.5-0.8B base + DSv4 additions.

    All dimensions are READ-ONLY after construction. Use `to_dict()` for serialization.
    """

    # ── Base architecture (from Qwen3.5-0.8B) ──────────────────────────────
    hidden_size: int = 1024
    num_hidden_layers: int = 24
    num_attention_heads: int = 8
    num_key_value_heads: int = 2    # GQA (not MQA)
    head_dim: int = 256
    intermediate_size: int = 3584
    vocab_size: int = 248320
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000000.0
    partial_rotary_factor: float = 0.25   # 0.25 * 256 = 64 dims RoPE'd
    attn_output_gate: bool = True
    tie_word_embeddings: bool = True
    mtp_num_hidden_layers: int = 1

    # ── DSv4 compression hyperparameters ───────────────────────────────────
    # CSA (Compressed Sparse Attention)
    csa_block_size: int = 4          # m - number of tokens per CSA block
    csa_compressed_dim: int = 256    # c - compressed KV dim per stream
    csa_intermediate: int = 512      # d_c - intermediate dim
    csa_groups: int = 4              # g - number of output groups
    csa_group_dim: int = 512         # d_g - group dim

    # HCA (Heavy Compressed Attention)
    hca_block_size: int = 128        # m' - tokens per HCA block
    hca_compressed_dim: int = 512    # KV stream compressed dim

    # SWA (Sliding Window Attention)
    swa_window_size: int = 128       # n_win

    # Lightning Indexer
    indexer_heads: int = 8           # n_Ih
    indexer_dim: int = 128           # c_I
    indexer_top_k: int = 128         # top-k blocks to retrieve

    # ── Derived block alignment ────────────────────────────────────────────
    # Block size = lcm(csa_block_size, hca_block_size) = lcm(4, 128) = 128
    block_alignment: int = 128

    # ── Rope ───────────────────────────────────────────────────────────────
    rope_dim: int = 64               # partial_rotary_factor * head_dim

    # ── Training ───────────────────────────────────────────────────────────
    lr_warmup: float = 1e-4
    lr_warmup_end: float = 1e-6
    lr_finetune: float = 2e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8

    # ── LoRA ───────────────────────────────────────────────────────────────
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")

    # ── Quantization ───────────────────────────────────────────────────────
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"

    # ── Layer type assignment (24 layers, 0-indexed) ──────────────────────
    # SWA: sliding window (uncompressed)
    # CSA: compressed sparse attention
    # HCA: heavy compressed attention
    swa_layers: tuple[int, ...] = (0, 1)
    csa_layers: tuple[int, ...] = (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
    hca_layers: tuple[int, ...] = (3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23)

    # ── Serialization helpers ──────────────────────────────────────────────
    _BASE_MODEL_PATH: ClassVar[str] = "Qwen/Qwen3.5-0.8B"

    def __post_init__(self) -> None:
        """Validate consistency."""
        # Layer count
        n_swa = len(self.swa_layers)
        n_csa = len(self.csa_layers)
        n_hca = len(self.hca_layers)
        assert n_swa + n_csa + n_hca == self.num_hidden_layers, (
            f"Layer assignment mismatch: {n_swa}+{n_csa}+{n_hca} != {self.num_hidden_layers}"
        )

        # Block alignment
        import math
        expected_block = abs(self.csa_block_size * self.hca_block_size) // math.gcd(
            self.csa_block_size, self.hca_block_size
        )
        assert self.block_alignment == expected_block, (
            f"block_alignment {self.block_alignment} != lcm({self.csa_block_size}, {self.hca_block_size}) = {expected_block}"
        )

        # Head dim must be divisible by rope partial factor
        assert int(self.head_dim * self.partial_rotary_factor) == self.rope_dim

        # Groups must divide heads
        assert self.num_attention_heads % self.csa_groups == 0, (
            f"csa_groups {self.csa_groups} does not divide heads {self.num_attention_heads}"
        )

        # Indexer heads match attention heads
        assert self.indexer_heads == self.num_attention_heads

    @classmethod
    def from_base_model(cls) -> "DSV4TinyConfig":
        """Create config by reading Qwen3.5-0.8B HuggingFace config."""
        try:
            from transformers import AutoConfig
        except ImportError:
            return cls()  # fallback to defaults

        hf_config = AutoConfig.from_pretrained(cls._BASE_MODEL_PATH)
        text_config = hf_config.text_config if hasattr(hf_config, "text_config") else hf_config

        hidden_size = getattr(text_config, "hidden_size", 1024)
        num_attention_heads = getattr(text_config, "num_attention_heads", 8)
        num_key_value_heads = getattr(text_config, "num_key_value_heads", 2)
        head_dim = getattr(text_config, "head_dim", 256)
        partial_rotary = getattr(text_config, "partial_rotary_factor", 0.25)
        rope_theta = getattr(text_config, "rope_theta", 10000000.0)
        intermediate_size = getattr(text_config, "intermediate_size", 3584)
        vocab_size = getattr(text_config, "vocab_size", 248320)
        max_pos = getattr(text_config, "max_position_embeddings", 262144)
        rms_norm_eps = getattr(text_config, "rms_norm_eps", 1e-6)
        attn_output_gate = getattr(text_config, "attn_output_gate", True)
        tie_weights = getattr(text_config, "tie_word_embeddings", True)
        num_layers = getattr(text_config, "num_hidden_layers", 24)

        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            vocab_size=vocab_size,
            max_position_embeddings=max_pos,
            rms_norm_eps=rms_norm_eps,
            rope_theta=rope_theta,
            partial_rotary_factor=partial_rotary,
            attn_output_gate=attn_output_gate,
            tie_word_embeddings=tie_weights,
            rope_dim=int(head_dim * partial_rotary),
            # DSv4 derived
            csa_compressed_dim=max(128, hidden_size // 4),   # 256
            csa_intermediate=max(128, hidden_size // 2),     # 512
            indexer_dim=128,
        )

    def layer_type(self, layer_idx: int) -> Literal["swa", "csa", "hca"]:
        if layer_idx in self.swa_layers:
            return "swa"
        elif layer_idx in self.csa_layers:
            return "csa"
        elif layer_idx in self.hca_layers:
            return "hca"
        raise ValueError(f"Layer {layer_idx} not assigned to any type")

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def __repr__(self) -> str:
        return (
            f"DSV4TinyConfig(hidden_size={self.hidden_size}, layers={self.num_hidden_layers}, "
            f"heads={self.num_attention_heads}, kv_heads={self.num_key_value_heads}, "
            f"head_dim={self.head_dim})"
        )
