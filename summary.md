# DSV4-Tiny Model Architecture

## Overview

**DSV4-Tiny** is a ~0.8B parameter causal language model implementing **DeepSeek-V4's three-tier heterogeneous KV cache** — a research prototype combining **Sliding Window Attention (SWA)**, **Compressed Sparse Attention (CSA)**, and **Heavy Compressed Attention (HCA)**. It is derived from **Qwen3.5-0.8B**, replacing all self-attention layers with DSV4's tiered attention while keeping the embedding, MLP, and output layers intact.

The goal: dramatically reduce KV cache memory at long contexts (up to 262k tokens) by compressing KV pairs into a hierarchy of storage tiers.

---

## Base Architecture (from Qwen3.5-0.8B)

| Hyperparameter | Value | Notes |
|---|---|---|
| `hidden_size` | 1024 | Model dimension |
| `num_hidden_layers` | 24 | Decoder layers |
| `num_attention_heads` | 8 | (n_h) |
| `num_key_value_heads` | 2 | GQA, 4 heads per KV head |
| `head_dim` | 256 | Per-head dimension |
| `intermediate_size` | 3584 | SwiGLU FFN hidden dim |
| `vocab_size` | 248,320 | Qwen3.5 tokenizer |
| `max_position_embeddings` | 262,144 | |
| `partial_rotary_factor` | 0.25 | Only last 64 dims are RoPE'd |
| `rms_norm_eps` | 1e-6 | |
| `rope_theta` | 10,000,000.0 | |
| `attn_output_gate` | true | Element-wise gate on output |
| `tie_word_embeddings` | true | Embedding / LM head weight tying |

### Layer Components (per decoder layer)

```
input_layernorm (RMSNorm)
    ↓
DSV4Attention (SWA / CSA / HCA)
    ↓  residual +
post_attention_layernorm (RMSNorm)
    ↓
SwiGLU MLP (gate_proj, up_proj, down_proj)
    ↓  residual +
```

- **RMSNorm**: Root Mean Square layer normalization.
- **SwiGLU MLP**: `SiLU(gate_proj(x)) * up_proj(x)` projected down by `down_proj`.

---

## Three-Tier Heterogeneous KV Cache

The 24 layers are partitioned into three attention types (Eq 1 from the DSv4 paper):

| Tier | Type | Layers | Storage | Block Size |
|---|---|---|---|---|
| 1 | **SWA** (Sliding Window) | 0, 1 | Uncompressed ring buffer | N/A (window = 128 tokens) |
| 2 | **CSA** (Compressed Sparse) | 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22 | Compressed blocks, retrieved via sparse index | 4 tokens per sub-block |
| 3 | **HCA** (Heavy Compressed) | 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23 | Compressed blocks, dense retrieval | 128 tokens per block |

### Block Alignment

- Block size = **128 tokens** = `lcm(CSA_block=4, HCA_block=128)`
- Each aligned block = **1 HCA block** = **32 CSA sub-blocks**

### Tier 1 — SWA (Sliding Window Attention)

- Layers 0 and 1 use **full uncompressed attention** over a sliding window of 128 tokens.
- K and V projections are computed fresh from hidden states (standard QKV).
- A ring buffer (`StateBuffer`) stores the last `window_size` KV pairs for fast access.
- Maintains an uncompressed tail for tokens not yet committed to CSA/HCA blocks.

### Tier 2 — CSA (Compressed Sparse Attention)

- 11 layers. KV is **compressed** before storage, grouping every 4 tokens into a block via `CSACompressor`.
- Retrieval is **sparse**: a `LightningIndexer` scores all compressed blocks and selects the top-K (default 128) most relevant blocks per query.
- Compressed block format (per sub-block of 4 tokens):
  - Content streams: **g groups × c_dim** = 4 × 256 = 1024 dims (dual-stream content)
  - Score stream: **d_c** = 512 (intermediate dim)
  - **Indexer key** (`KI_Comp`): used by the LightningIndexer for scoring
- Total per sub-block: 1024 + 512 = 1536 dims
- Per aligned block (32 sub-blocks): 1536 × 32 = 49,152 dims (vs ~131,072 uncompressed)

### Tier 3 — HCA (Heavy Compressed Attention)

- 11 layers. KV is **compressed** into a single vector per 128-token aligned block via `HCACompressor`.
- Retrieval is **dense**: all available blocks are attended over.
- Compressed block format: **c_hca** = 512 dims (single stream, no gating).
- Per aligned block: 512 dims (vs ~131,072 uncompressed — ~256× compression).

---

## Attention Logic (DSV4Attention)

Each `DSV4Attention` layer:

1. **Q projection**: `q_proj(hidden_states)` → (b, seq_len, n_h × head_dim)
2. **Partial RoPE**: Apply rotary embeddings to the last `rope_dim` (64) dims of Q
3. **Route by layer type**:
   - **SWA**: Compute K, V from projections, do standard MHA/GQA with a causal sliding window mask. KV pairs stored/retrieved from the `StateBuffer` ring buffer.
   - **CSA**: Decompress top-K blocks from Tier 2 cache, apply sparse attention:
     1. Decomposed query `c_Q = W_DQ(h)` (shared with indexer)
     2. Up-projected query `q = W_UQ(c_Q)` per head
     3. Indexer retrieves top-K block indices
     4. Decompress selected blocks via `kv_decompress`
     5. GQA attention over selected blocks
     6. Output gated through `o_gate`
   - **HCA**: Attend over all Tier 3 compressed blocks:
     1. Decompress all HCA blocks via `kv_decompress`
     2. GQA attention over compressed blocks
     3. Output gated through `o_gate`

### Attention Sink Token (Eq 27)

- A learnable `sink_token` (1 × 1 × head_dim) shared across layers, prepended to KV sequences for stability — same mechanism used in StreamingLLM and Microsoft's LongRoPE.

### Output Gate

- `attn_output_gate = True`: the attention output `x` is element-wise gated by `sigmoid(o_gate(x)) * x`, a gating mechanism from modern linear attention.

---

## Compression Modules

### CSACompressor (Eq 9–12)

Compresses 4 tokens into a single block using **dual-stream content pooling**:

```
Content streams:   C_a = H @ W_aKV      (content stream a)
                   C_b = H @ W_bKV      (content stream b)
Score streams:     Z_a = H @ W_aZ + B_a (position-aware scores a)
                   Z_b = H @ W_bZ + B_b (position-aware scores b)
Weights:           [S_a; S_b] = Softmax([Z_a; Z_b], dim=-2)
Output:            C = sum(S_a * C_a) + sum(S_b * C_b)
```

- g = 4 output groups, each producing 256-dim compressed KV
- Intermediate score dim d_c = 512
- `W_aKV`, `W_bKV`: (4 × 1024) → (4 × c × g) = (4, 1024, 1024) each
- `W_aZ`, `W_bZ`: (4, 1024, d_c) = (4, 1024, 512) each
- Biases `B_a`, `B_b`: (4, d_c) initialized to zeros

### HCACompressor (Eq 20–23)

Compresses 128 tokens into a single KV vector:

```
Content stream: C = H @ W_KV    (dual-stream: 2 × c_hca = 1024 dims)
Score stream:   Z = H @ W_Z + B (position-aware scores)
Weight:         S = Softmax(Z, dim=-2)
Output:         C_comp = sum(S[:, :, :c_hca] * C[:, :, :c_hca]) +
                          sum(S[:, :, c_hca:] * C[:, :, c_hca:])
```

- c_hca = 512 compression dim (dual 1024 total)
- `W_KV`: (128, 1024, 1024)
- `W_Z`: (128, 1024, 512)
- `B`: (128, 512) initialized to zeros

---

## LightningIndexer (Eq 13–17)

Lightweight neural indexer for **sparse block retrieval** (used by CSA layers):

```
Decomposed query:      c_Q = W_DQ(h)        shape: (b, seq, c_I × n_Ih)
Indexer query:         q_I = W_IUQ(c_Q)     per-head up-projection
Head-wise score:       w_I = W_w(h)         scalar per head per position
Per-block score:       I_s = sum_h[w_I_h · ReLU(q_I_h @ KI_Comp_s)]
```

- `c_I` = 128 (indexer dim)
- `n_Ih` = 8 (indexer heads, matching attention heads)
- `top_k` = 128 (blocks retrieved per query position)
- `W_DQ`: (1024, 1024) → decomposed query shared with attention
- `W_IUQ`: (128, 128) per-head up-projection
- `W_w`: (1024, 8) head-wise scoring weights

The indexer computes scores against every stored `KI_Comp` (indexer key) in the CSA cache, selects the top-K blocks, and returns their indices for sparse attention.

---

## KV Cache (DSV4Cache)

The `DSV4Cache` module manages all three tiers:

- **StateBuffer**: Ring buffer holding `window_size` (128) uncompressed KV pairs for SWA layers
- **CSA block list**: Ordered list of `CompressedBlock` objects (content + indexer key)
- **HCA block list**: Ordered list of compressed blocks (content only)

**Commit strategy**: As new KV pairs arrive:
1. Serve SWA layers from the ring buffer
2. When `csa_block_size` (4) tokens accumulate, compress into a CSA block and store
3. When `hca_block_size` (128) tokens accumulate, compress into an HCA block and store
4. Uncompressed tail (< block_alignment tokens) stored temporarily until commit

**Estimated cache size** (for 128k tokens, FP16):
- Uncompressed baseline: ~12 GiB
- DSV4 compressed: ~1.3 GiB (~9× reduction)
  - SWA: 2 layers × 128 tokens × dims
  - CSA: 11 layers × 32,768 blocks × 1,536 dims
  - HCA: 11 layers × 1,024 blocks × 512 dims
  - Tail: 2 layers × 128 tokens × dims

---

## Weight Loading (`from_pretrained`)

`DSV4TinyForCausalLM.from_pretrained("Qwen/Qwen3.5-0.8B")`:

1. Reads Qwen3.5-0.8B HuggingFace config → creates `DSV4TinyConfig`
2. Creates model skeleton (random init)
3. Loads base model via `AutoModelForCausalLM`
4. Copies: embeddings, RMSNorm weights, MLP weights, O projection, LM head
5. Copies Q projection (or composites from GatedDeltaNet if present)
6. Copies K/V projections for SWA layers only (layers 0, 1)
7. Compression/indexer weights remain randomly initialized (N(0, 0.02))

---

## Parameter Count (Estimated)

| Component | Parameters |
|---|---|
| Embedding (tied) | 248,320 × 1024 = 254.3M |
| Q projection (24 layers) | 24 × 1024 × 2048 = 50.3M |
| K/V projections (2 SWA layers) | 2 × 1024 × 256 = 0.5M |
| KV decompress (22 compressed layers) | 22 × (CSA/HCA dim × 512) ≈ 21–35M |
| O projection (24 layers) | 24 × 2048 × 1024 = 50.3M |
| O gate (24 layers) | 24 × 2048 × 1024 = 50.3M |
| MLP (24 × 3 projections) | 24 × (1024×3584 × 2 + 3584×1024) = 264.2M |
| Compression modules | ~5–8M |
| Indexer | ~1M |
| Misc (norms, sink) | ~0.1M |
| **Total** | **~700–750M** |

*Note: exact count depends on CSA/HCA decompression dims; embedding is the dominant cost.*

---

## Training Pipeline

### Phase 1 — Warm-start (scripts/warm_start.py)

Trains **only the new DSv4 weights** (compressors, indexer, WV_DQ, WV_UQ) while freezing base model weights. Uses an **autoencoder-style loss**: compress hidden states, decompress, and compare to originals. Trained on synthetic random token data with a cosine decay LR schedule (1e-4 → 1e-6).

### Phase 2 — LoRA Fine-tuning (scripts/finetune_lora.py)

Full fine-tuning via `peft` LoRA (r=16, alpha=16) on `q_proj`, `k_proj`, `v_proj`, `o_proj`. Uses `unsloth` for 4-bit quantization (NF4, bfloat16 compute) and memory-efficient training. Default LR = 2e-4 with cosine decay.

### Inference Server (src/dsv4_tiny/inference.py)

FastAPI server with OpenAI-compatible `/v1/chat/completions` endpoint. Loads the model, provides streaming and non-streaming generation.

---

## Summary Diagram

```
Input Tokens
    │
    ▼
 Embedding (1024d)
    │
    ▼
  ┌──────────────────────────────────────────────────────┐
  │  24× Decoder Layers                                  │
  │                                                      │
  │  Layer 0-1:  SWA (Sliding Window, window=128)        │
  │    └─ K/V from projections, ring buffer cache        │
  │                                                      │
  │  Layer 2,4,...,22:  CSA (Compressed Sparse, block=4) │
  │    └─ Compressed KV + LightningIndexer (top-K=128)   │
  │                                                      │
  │  Layer 3,5,...,23:  HCA (Heavy Compressed, block=128)│
  │    └─ High-compression KV, dense retrieval           │
  │                                                      │
  │  Each layer: Norm → DSV4Attention → Norm → SwiGLU MLP│
  └──────────────────────────────────────────────────────┘
    │
    ▼
 RMSNorm → LM Head → Output Logits
```

### Cache Hierarchy

```
                    ┌──────────────────────┐
                    │   Tier 1: SWA Buffer  │  (ring buffer, 128 tokens)
                    │   Layers 0, 1        │
                    └──────────┬───────────┘
                               │
             4 tokens ────────┼────────────
                    ┌──────────▼───────────┐
                    │   Tier 2: CSA Blocks  │  (4-token sub-blocks)
                    │   Layers 2,4,...,22   │  ← LightningIndexer
                    │   top-K = 128/block   │     (sparse retrieval)
                    └──────────┬───────────┘
                               │
            128 tokens ───────┼────────────
                    ┌──────────▼───────────┐
                    │   Tier 3: HCA Blocks  │  (128-token blocks)
                    │   Layers 3,5,...,23   │  (dense retrieval)
                    └──────────────────────┘
```
