# Complete Layer Shapes & Sizes Analysis
## Configuration: hidden_size=768, num_layers=12, batch_size=32

---

## Model Configuration

```python
vocab_size = 32000         # From your data preparation
hidden_size = 768          # Your config
num_layers = 12            # Your config
num_heads = 12             # Default
num_kv_heads = 12          # Default (same as num_heads)
max_seq_length = 2048      # Default
batch_size = 32            # Your config
intermediate_size = 3072   # Default (4 × hidden_size)
```

---

## Input Stage

### Initial Input

**Token IDs (from dataloader):**
```
Shape: [batch_size, seq_len]
Shape: [32, 2048]
Type: torch.LongTensor (int64)
Size: 32 × 2048 = 65,536 token IDs
Memory: 65,536 × 8 bytes = 524 KB
```

**Example batch:**
```python
input_ids = [
    [15, 496, 995, 67, 89, ..., 2],  # Sentence 1 (2048 tokens)
    [23, 891, 234, 12, 45, ..., 2],  # Sentence 2 (2048 tokens)
    ...
    [78, 234, 567, 89, 90, ..., 2],  # Sentence 32 (2048 tokens)
]
```

---

## Layer 0: Embedding Layer

### Token Embedding

**Operation:** Look up embeddings for each token ID

**Parameters:**
```
Weight matrix: [vocab_size, hidden_size]
Weight matrix: [32000, 768]
Parameters: 32,000 × 768 = 24,576,000
Memory: 24,576,000 × 4 bytes (FP32) = 98.3 MB
Memory: 24,576,000 × 2 bytes (FP16) = 49.2 MB
```

**Output:**
```
Shape: [batch_size, seq_len, hidden_size]
Shape: [32, 2048, 768]
Values: 32 × 2048 × 768 = 50,331,648 values
Memory (FP16): 50,331,648 × 2 bytes = 96.5 MB
```

**Visual representation:**
```
Input:  [32, 2048]      → 32 sentences, 2048 token IDs each
         ↓
Embedding lookup in [32000, 768] table
         ↓
Output: [32, 2048, 768] → 32 sentences, 2048 vectors of 768 dimensions
```

---

## Transformer Block (Repeated 12 times)

### Block Structure Overview

```
Input: [32, 2048, 768]
    ↓
┌─────────────────────────────────────┐
│  Pre-Attention Norm (RMSNorm)       │
│  Input:  [32, 2048, 768]            │
│  Output: [32, 2048, 768]            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Grouped Query Attention            │
│  Input:  [32, 2048, 768]            │
│  Output: [32, 2048, 768]            │
└─────────────────────────────────────┘
    ↓
Residual Connection (+)
    ↓
┌─────────────────────────────────────┐
│  Pre-FFN Norm (RMSNorm)             │
│  Input:  [32, 2048, 768]            │
│  Output: [32, 2048, 768]            │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│  Feed-Forward (SwiGLU)              │
│  Input:  [32, 2048, 768]            │
│  Output: [32, 2048, 768]            │
└─────────────────────────────────────┘
    ↓
Residual Connection (+)
    ↓
Output: [32, 2048, 768]
```

---

## Layer 1-12: Detailed Breakdown

### Step 1: RMSNorm (Pre-Attention)

**Parameters:**
```
Weight (gamma): [hidden_size]
Weight (gamma): [768]
Parameters: 768
Memory: 768 × 4 bytes = 3 KB
```

**Operation:**
```python
# Compute RMS
rms = sqrt(mean(x^2) + eps)
# Normalize
x_norm = x / rms
# Scale
output = x_norm * weight
```

**Shapes:**
```
Input:  [32, 2048, 768]
Output: [32, 2048, 768]  ← Same shape
```

---

### Step 2: Grouped Query Attention

#### 2a. Project to Q, K, V

**Query Projection:**
```
Weight: [hidden_size, hidden_size]
Weight: [768, 768]
Parameters: 768 × 768 = 589,824
Memory: 589,824 × 2 bytes (FP16) = 1.15 MB

Input:  [32, 2048, 768]
Output: [32, 2048, 768]  ← Q (queries)
```

**Key Projection:**
```
Weight: [hidden_size, num_kv_heads × head_dim]
Weight: [768, 768]  (since num_kv_heads = num_heads = 12)
Parameters: 768 × 768 = 589,824
Memory: 589,824 × 2 bytes = 1.15 MB

Input:  [32, 2048, 768]
Output: [32, 2048, 768]  ← K (keys)
```

**Value Projection:**
```
Weight: [768, 768]
Parameters: 589,824
Memory: 1.15 MB

Input:  [32, 2048, 768]
Output: [32, 2048, 768]  ← V (values)
```

#### 2b. Reshape for Multi-Head

**Head dimensions:**
```
head_dim = hidden_size / num_heads
head_dim = 768 / 12 = 64
```

**Reshape Q, K, V:**
```
Before: [32, 2048, 768]
Split into heads: [32, 2048, 12, 64]
Transpose: [32, 12, 2048, 64]

Q: [batch, heads, seq_len, head_dim]
Q: [32, 12, 2048, 64]

K: [32, 12, 2048, 64]
V: [32, 12, 2048, 64]
```

#### 2c. Apply Rotary Position Embeddings (RoPE)

**Compute cos and sin:**
```
Frequencies shape: [2048, 64]
Cos: [1, 2048, 64]
Sin: [1, 2048, 64]
```

**Apply to Q and K:**
```
Q_rotated: [32, 12, 2048, 64]
K_rotated: [32, 12, 2048, 64]
(Same shape, values modified)
```

#### 2d. Compute Attention Scores

**Matrix multiplication:**
```
scores = Q @ K^T

Q: [32, 12, 2048, 64]
K^T: [32, 12, 64, 2048]  ← Transpose last 2 dims

Result: [32, 12, 2048, 2048]
```

**Attention scores shape:**
```
[batch, heads, seq_len, seq_len]
[32, 12, 2048, 2048]

Values: 32 × 12 × 2048 × 2048 = 1,610,612,736
Memory: 1,610,612,736 × 2 bytes = 3.2 GB!
```

**This is why Flash Attention is important!**
```
Standard attention: Materializes full [2048, 2048] matrix
Flash Attention: Processes in blocks, never materializes full matrix
Memory savings: ~10x
```

#### 2e. Scale Scores

```
scores = scores / sqrt(head_dim)
scores = scores / sqrt(64)
scores = scores / 8

Shape: [32, 12, 2048, 2048]  ← Unchanged
```

#### 2f. Apply Causal Mask

**Mask shape:**
```
[1, 1, 2048, 2048]  ← Broadcasted to all batches/heads

Mask matrix (upper triangle = -inf):
[[ 0,  -∞,  -∞,  -∞, ...],
 [ 0,   0,  -∞,  -∞, ...],
 [ 0,   0,   0,  -∞, ...],
 ...]

This prevents attending to future tokens
```

**After masking:**
```
Shape: [32, 12, 2048, 2048]
```

#### 2g. Softmax

```
attn_weights = softmax(scores, dim=-1)

Input:  [32, 12, 2048, 2048]
Output: [32, 12, 2048, 2048]
(Each row sums to 1.0)
```

#### 2h. Apply Attention to Values

**Matrix multiplication:**
```
output = attn_weights @ V

attn_weights: [32, 12, 2048, 2048]
V: [32, 12, 2048, 64]

Result: [32, 12, 2048, 64]
```

#### 2i. Concatenate Heads

**Reshape back:**
```
Input:  [32, 12, 2048, 64]
Transpose: [32, 2048, 12, 64]
Reshape: [32, 2048, 768]  ← Concatenate all heads
```

#### 2j. Output Projection

```
Weight: [768, 768]
Parameters: 589,824
Memory: 1.15 MB

Input:  [32, 2048, 768]
Output: [32, 2048, 768]
```

**Total Attention Parameters per layer:**
```
Q projection: 589,824
K projection: 589,824
V projection: 589,824
O projection: 589,824
Total: 2,359,296 parameters
Memory: 4.7 MB (FP16)
```

---

### Step 3: Residual Connection

```
attention_output: [32, 2048, 768]
original_input:   [32, 2048, 768]
                  +
result:           [32, 2048, 768]
```

---

### Step 4: RMSNorm (Pre-FFN)

```
Parameters: 768
Memory: 3 KB

Input:  [32, 2048, 768]
Output: [32, 2048, 768]
```

---

### Step 5: Feed-Forward Network (SwiGLU)

#### 5a. Gate Projection

```
Weight: [hidden_size, intermediate_size]
Weight: [768, 3072]
Parameters: 768 × 3072 = 2,359,296
Memory: 4.7 MB (FP16)

Input:  [32, 2048, 768]
Output: [32, 2048, 3072]  ← Expanded!
```

#### 5b. Up Projection

```
Weight: [768, 3072]
Parameters: 2,359,296
Memory: 4.7 MB

Input:  [32, 2048, 768]
Output: [32, 2048, 3072]
```

#### 5c. SwiGLU Activation

```
gate = SiLU(gate_proj(x))
up = up_proj(x)
activated = gate * up  ← Element-wise multiplication

gate:   [32, 2048, 3072]
up:     [32, 2048, 3072]
result: [32, 2048, 3072]
```

#### 5d. Down Projection

```
Weight: [intermediate_size, hidden_size]
Weight: [3072, 768]
Parameters: 3072 × 768 = 2,359,296
Memory: 4.7 MB

Input:  [32, 2048, 3072]
Output: [32, 2048, 768]  ← Back to hidden_size
```

**Total FFN Parameters per layer:**
```
Gate projection: 2,359,296
Up projection:   2,359,296
Down projection: 2,359,296
Total: 7,077,888 parameters
Memory: 14.2 MB (FP16)
```

---

### Step 6: Residual Connection

```
ffn_output:     [32, 2048, 768]
original_input: [32, 2048, 768]
                +
result:         [32, 2048, 768]
```

---

## Complete Transformer Block Summary

**Per-Block Parameters:**
```
RMSNorm (pre-attention): 768
Attention: 2,359,296
RMSNorm (pre-FFN): 768
Feed-Forward: 7,077,888
──────────────────────────
Total per block: 9,438,720 parameters
Memory per block: 18.9 MB (FP16)
```

**Shapes through one block:**
```
Input:           [32, 2048, 768]
↓ RMSNorm        [32, 2048, 768]
↓ Attention      [32, 2048, 768]
↓ Residual       [32, 2048, 768]
↓ RMSNorm        [32, 2048, 768]
↓ FFN            [32, 2048, 768]
↓ Residual       [32, 2048, 768]
Output:          [32, 2048, 768]
```

---

## All 12 Transformer Blocks

**Stacked processing:**
```
Block 0:  [32, 2048, 768] → [32, 2048, 768]
Block 1:  [32, 2048, 768] → [32, 2048, 768]
Block 2:  [32, 2048, 768] → [32, 2048, 768]
Block 3:  [32, 2048, 768] → [32, 2048, 768]
Block 4:  [32, 2048, 768] → [32, 2048, 768]
Block 5:  [32, 2048, 768] → [32, 2048, 768]
Block 6:  [32, 2048, 768] → [32, 2048, 768]
Block 7:  [32, 2048, 768] → [32, 2048, 768]
Block 8:  [32, 2048, 768] → [32, 2048, 768]
Block 9:  [32, 2048, 768] → [32, 2048, 768]
Block 10: [32, 2048, 768] → [32, 2048, 768]
Block 11: [32, 2048, 768] → [32, 2048, 768]
```

**Total Transformer Parameters:**
```
12 blocks × 9,438,720 = 113,264,640 parameters
Memory: 226.5 MB (FP16)
```

---

## Final Layers

### Final RMSNorm

```
Parameters: 768
Memory: 3 KB

Input:  [32, 2048, 768]
Output: [32, 2048, 768]
```

### Output Projection (LM Head)

**Linear layer:**
```
Weight: [hidden_size, vocab_size]
Weight: [768, 32000]
Parameters: 768 × 32,000 = 24,576,000
Memory: 49.2 MB (FP16)

Input:  [32, 2048, 768]
Output: [32, 2048, 32000]  ← Vocabulary logits
```

**Final output:**
```
Shape: [batch_size, seq_len, vocab_size]
Shape: [32, 2048, 32000]

This represents:
- 32 sentences
- 2048 positions per sentence
- 32000 possible next tokens at each position
```

---

## Loss Computation

### Shift for Next-Token Prediction

```
Logits: [32, 2048, 32000]
Labels: [32, 2048]

# Shift (predict next token)
shift_logits = logits[:, :-1, :]  # [32, 2047, 32000]
shift_labels = labels[:, 1:]       # [32, 2047]

# Flatten for cross-entropy
shift_logits = shift_logits.reshape(-1, 32000)  # [65504, 32000]
shift_labels = shift_labels.reshape(-1)          # [65504]
```

### Cross-Entropy Loss

```
Input logits: [65504, 32000]
Input labels: [65504]

For each of 65,504 positions:
- Convert logits to probabilities (softmax)
- Compute: loss = -log(P(correct_token))

Output: scalar loss value
```

---

## Complete Model Summary

### Total Parameters

```
Embedding:          24,576,000
Transformer blocks: 113,264,640  (12 blocks × 9,438,720)
Final norm:         768
LM head:            24,576,000
──────────────────────────────────
Total:              162,417,408 parameters

In millions: ~162M parameters
Memory (FP16): 324 MB
Memory (FP32): 648 MB
```

### Parameter Breakdown by Component

```
┌────────────────────┬──────────────┬─────────┐
│ Component          │ Parameters   │ Percent │
├────────────────────┼──────────────┼─────────┤
│ Embeddings         │ 24,576,000   │ 15.1%   │
│ Attention (all)    │ 28,311,552   │ 17.4%   │
│ Feed-Forward (all) │ 84,934,656   │ 52.3%   │
│ Norms (all)        │ 9,984        │ 0.006%  │
│ LM Head            │ 24,576,000   │ 15.1%   │
└────────────────────┴──────────────┴─────────┘
```

---

## Memory Breakdown During Training

### Forward Pass Memory

```
Activations per layer (approximate):
- Hidden states: [32, 2048, 768] = 50.3M values
- Attention scores: [32, 12, 2048, 2048] = 1,611M values (with Flash Attn: much less)
- Intermediate FFN: [32, 2048, 3072] = 201M values

Total activation memory (without Flash Attention):
~10-15 GB per batch

With Flash Attention:
~2-3 GB per batch
```

### Backward Pass Memory

```
Gradients (same size as parameters):
FP16: 324 MB

Optimizer states (Adam):
- First moment: 324 MB
- Second moment: 324 MB
Total: 648 MB

Total training memory:
Model: 324 MB
Gradients: 324 MB
Optimizer: 648 MB
Activations: 2-3 GB (with Flash Attention)
──────────────────
Total: ~4-5 GB minimum
```

### With Mixed Precision

```
Model weights: FP16 (324 MB)
Master weights: FP32 (648 MB)
Gradients: FP16 (324 MB)
Optimizer: FP32 (1,296 MB)
Activations: FP16 (2-3 GB)
────────────────────────────
Total: ~5-6 GB

This fits comfortably on:
- RTX 3060 (12GB) ✓
- RTX 3070 (8GB) ✓
- RTX 3080 (10GB) ✓
- RTX 3090 (24GB) ✓✓
```

---

## Data Flow Visualization

```
Step-by-Step:

1. Input Token IDs
   [32, 2048] (int64)
   
2. Embedding Lookup
   [32, 2048, 768] (FP16)
   
3. 12× Transformer Blocks
   Each: [32, 2048, 768] → [32, 2048, 768]
   
4. Final Norm
   [32, 2048, 768] (FP16)
   
5. LM Head Projection
   [32, 2048, 32000] (FP16)
   
6. Loss Computation
   Scalar value (FP32)
```

---

## Shape Tracking Code

You can add this to your training script to verify shapes:

```python
import torch

def print_shapes(name, tensor):
    """Helper to print tensor shapes during training"""
    print(f"{name:30s}: {list(tensor.shape):20s} | {tensor.dtype} | {tensor.numel():,} values")

# In your forward pass:
print_shapes("Input IDs", input_ids)
print_shapes("Embedded", embedded)

for i, layer in enumerate(self.layers):
    hidden_states = layer(hidden_states)
    if i % 3 == 0:  # Print every 3rd layer
        print_shapes(f"After Block {i}", hidden_states)

print_shapes("After Final Norm", normalized)
print_shapes("Logits", logits)
print_shapes("Loss", loss)
```

---

## Quick Reference Table

| Layer | Input Shape | Output Shape | Parameters |
|-------|-------------|--------------|------------|
| Input | [32, 2048] | [32, 2048] | 0 |
| Embedding | [32, 2048] | [32, 2048, 768] | 24.6M |
| Block 0-11 | [32, 2048, 768] | [32, 2048, 768] | 9.4M each |
| Final Norm | [32, 2048, 768] | [32, 2048, 768] | 768 |
| LM Head | [32, 2048, 768] | [32, 2048, 32000] | 24.6M |
| **Total** | - | - | **162.4M** |

This model is similar in size to GPT-2 Small (124M) but with modern improvements like RoPE, SwiGLU, and RMSNorm!