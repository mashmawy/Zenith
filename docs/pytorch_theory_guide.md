# PyTorch Knowledge & Theory Guide
## Understanding the LLM Training Framework Code

---

## Table of Contents
1. [PyTorch Fundamentals](#pytorch-fundamentals)
2. [Neural Network Basics](#neural-network-basics)
3. [Transformer Architecture Theory](#transformer-architecture-theory)
4. [Training Process Deep Dive](#training-process-deep-dive)
5. [Distributed Training Concepts](#distributed-training-concepts)
6. [Code Walkthrough with Theory](#code-walkthrough-with-theory)
7. [Mathematical Foundations](#mathematical-foundations)

---

## Part 1: PyTorch Fundamentals

### What is PyTorch?

PyTorch is a machine learning library that provides:
- **Tensors**: Multi-dimensional arrays (like NumPy, but GPU-enabled)
- **Autograd**: Automatic differentiation (calculates gradients)
- **Neural Network Modules**: Building blocks for models
- **Optimizers**: Algorithms to update model weights

### 1.1 Tensors - The Foundation

**What is a Tensor?**
A tensor is just a multi-dimensional array of numbers.

```python
import torch

# Scalar (0D tensor) - just a number
scalar = torch.tensor(5)
print(scalar.shape)  # torch.Size([])

# Vector (1D tensor) - list of numbers
vector = torch.tensor([1, 2, 3, 4])
print(vector.shape)  # torch.Size([4])

# Matrix (2D tensor) - table of numbers
matrix = torch.tensor([[1, 2], [3, 4], [5, 6]])
print(matrix.shape)  # torch.Size([3, 2]) - 3 rows, 2 columns

# 3D tensor - stack of matrices
tensor_3d = torch.tensor([
    [[1, 2], [3, 4]],
    [[5, 6], [7, 8]]
])
print(tensor_3d.shape)  # torch.Size([2, 2, 2])
```

**In Our Code:**
```python
# Input IDs shape: [batch_size, sequence_length]
input_ids = torch.tensor([[15, 496, 995], [89, 12, 67]])
# Shape: [2, 3] - 2 sentences, 3 tokens each

# Model output shape: [batch_size, sequence_length, vocab_size]
# Shape: [2, 3, 32000] - 2 sentences, 3 positions, 32000 possible next tokens
```

**Key Tensor Operations:**

```python
# Create tensors
x = torch.randn(3, 4)  # Random normal distribution, shape [3, 4]
y = torch.zeros(2, 5)  # All zeros
z = torch.ones(4, 4)   # All ones

# Move to GPU
x_gpu = x.to('cuda')   # If GPU available
x_cpu = x_gpu.to('cpu')  # Back to CPU

# Basic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])
c = a + b              # Element-wise: [5, 7, 9]
d = a * b              # Element-wise: [4, 10, 18]
e = torch.matmul(a, b) # Dot product: 32

# Reshaping
x = torch.randn(12)
y = x.view(3, 4)       # Reshape to [3, 4]
z = x.view(-1, 2)      # Reshape to [?, 2], infers first dim: [6, 2]

# Indexing
x = torch.randn(4, 5)
first_row = x[0]        # Get first row
first_col = x[:, 0]     # Get first column
subset = x[1:3, 2:4]    # Rows 1-2, columns 2-3
```

### 1.2 Autograd - Automatic Differentiation

**Why do we need it?**
Training neural networks requires computing gradients (derivatives) to know how to update weights.

**Basic Example:**

```python
import torch

# Create tensor with gradient tracking
x = torch.tensor([2.0], requires_grad=True)

# Forward pass - define computation
y = x ** 2  # y = x²

# Backward pass - compute gradient
y.backward()  # dy/dx = 2x

# Access gradient
print(x.grad)  # tensor([4.]) because 2 * 2 = 4
```

**In Neural Networks:**

```python
# Model parameters automatically track gradients
model = torch.nn.Linear(10, 5)
for param in model.parameters():
    print(param.requires_grad)  # True

# During training
output = model(input_data)
loss = loss_function(output, target)

# This computes gradients for ALL parameters
loss.backward()

# Access gradients
for param in model.parameters():
    print(param.grad)  # Gradients computed!
```

**In Our Code (train.py):**

```python
# Forward pass
logits, loss = self.model(input_ids, labels=labels)

# Backward pass - computes gradients automatically
loss.backward()

# Update weights using gradients
self.optimizer.step()  # Uses computed gradients

# Clear gradients for next iteration
self.optimizer.zero_grad()
```

### 1.3 Neural Network Modules (nn.Module)

**What is nn.Module?**
The base class for all neural network components. It:
- Manages parameters (weights)
- Tracks gradients
- Provides easy GPU transfer
- Enables saving/loading

**Basic Structure:**

```python
import torch.nn as nn

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()  # MUST call parent constructor
        
        # Define layers (these are also nn.Modules)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        # Define forward pass
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# Create model
model = SimpleNet(10, 20, 5)

# Use model
input_data = torch.randn(32, 10)  # Batch of 32
output = model(input_data)  # Calls forward() automatically
print(output.shape)  # [32, 5]
```

**In Our Code (model.py):**

```python
class LLMModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Embedding layer
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Stack of transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final normalization
        self.norm = RMSNorm(config.hidden_size)
        
        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids):
        # Embed tokens
        x = self.embed_tokens(input_ids)
        
        # Pass through transformer layers
        for layer in self.layers:
            x = layer(x)
        
        # Normalize and project to vocabulary
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
```

**Key Methods:**

```python
# Get all parameters
for name, param in model.named_parameters():
    print(name, param.shape)

# Move to GPU
model = model.to('cuda')

# Set to training/evaluation mode
model.train()  # Enables dropout, batch norm training
model.eval()   # Disables dropout, batch norm eval

# Save/load
torch.save(model.state_dict(), 'model.pt')
model.load_state_dict(torch.load('model.pt'))
```

### 1.4 Common Layers

**Linear Layer (Fully Connected):**

```python
# y = xW^T + b
linear = nn.Linear(in_features=10, out_features=5, bias=True)

x = torch.randn(32, 10)  # Batch of 32, 10 features
y = linear(x)            # Output: [32, 5]

# Parameters
print(linear.weight.shape)  # [5, 10]
print(linear.bias.shape)    # [5]
```

**Embedding Layer:**

```python
# Converts token IDs to dense vectors
embedding = nn.Embedding(num_embeddings=1000, embedding_dim=128)

# Input: token IDs
input_ids = torch.tensor([[1, 5, 2], [3, 8, 9]])  # [2, 3]

# Output: dense vectors
embedded = embedding(input_ids)  # [2, 3, 128]
# Each token ID is now a 128-dimensional vector
```

**Layer Normalization:**

```python
# Normalizes across feature dimension
layer_norm = nn.LayerNorm(normalized_shape=768)

x = torch.randn(32, 10, 768)  # [batch, seq, features]
normalized = layer_norm(x)    # Same shape, but normalized

# Makes training more stable
```

**Dropout:**

```python
# Randomly zeros elements (prevents overfitting)
dropout = nn.Dropout(p=0.1)  # Drop 10% of values

x = torch.randn(32, 768)
x_dropped = dropout(x)  # 10% of values are 0

# Only active in training mode
model.train()  # Dropout active
model.eval()   # Dropout disabled
```

---

## Part 2: Neural Network Basics

### 2.1 What is a Neural Network?

**Simple Explanation:**
A neural network is a function that:
1. Takes input (numbers)
2. Transforms it through layers
3. Produces output (predictions)

**Mathematical View:**
```
Input → Layer 1 → Layer 2 → ... → Layer N → Output

Each layer: y = activation(Wx + b)
Where:
- W = weights (learned parameters)
- b = bias (learned parameters)
- x = input
- activation = non-linear function
```

**Example - Predicting House Prices:**

```python
class HousePriceModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Input: 3 features (size, bedrooms, age)
        # Hidden: 10 neurons
        # Output: 1 (price)
        self.layer1 = nn.Linear(3, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x shape: [batch, 3] - 3 features
        x = self.layer1(x)      # [batch, 10]
        x = self.relu(x)        # [batch, 10] - non-linearity
        x = self.layer2(x)      # [batch, 1] - price prediction
        return x

# Use it
model = HousePriceModel()
house = torch.tensor([[2000, 3, 5]])  # 2000 sqft, 3 bed, 5 years old
price = model(house)
print(f"Predicted price: ${price.item():,.0f}")
```

### 2.2 Activation Functions

**Why do we need them?**
Without activation functions, stacking layers is useless - it's just linear transformations.

**Common Activations:**

```python
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU: max(0, x)
relu = F.relu(x)
# Output: [0, 0, 0, 1, 2]

# Sigmoid: 1 / (1 + e^-x)
sigmoid = torch.sigmoid(x)
# Output: [0.12, 0.27, 0.5, 0.73, 0.88]

# Tanh: (e^x - e^-x) / (e^x + e^-x)
tanh = torch.tanh(x)
# Output: [-0.96, -0.76, 0, 0.76, 0.96]

# SiLU (Swish): x * sigmoid(x)
silu = F.silu(x)
# Output: [-0.24, -0.27, 0, 0.73, 1.76]

# Softmax: e^xi / sum(e^xj)
softmax = F.softmax(x, dim=0)
# Output: [0.01, 0.03, 0.09, 0.24, 0.63] - sums to 1
```

**In Our Code:**
```python
# SwiGLU activation (advanced)
def forward(self, x):
    gate = F.silu(self.gate_proj(x))  # Swish activation
    up = self.up_proj(x)
    return self.down_proj(gate * up)  # Gated combination
```

### 2.3 Loss Functions

**What is Loss?**
A measure of how wrong the model's predictions are.

**Cross-Entropy Loss (for classification):**

```python
import torch.nn.functional as F

# Model outputs (logits)
logits = torch.tensor([[2.0, 1.0, 0.1]])  # 3 classes

# True label
target = torch.tensor([0])  # Correct class is 0

# Compute loss
loss = F.cross_entropy(logits, target)
print(loss)  # Higher if prediction is wrong

# What it does:
# 1. Convert logits to probabilities (softmax)
# 2. Take negative log of correct class probability
# 3. Lower probability → higher loss
```

**In Language Models:**

```python
# Predict next token
logits = model(input_ids)  # [batch, seq, vocab_size]

# Shift for next-token prediction
shift_logits = logits[:, :-1, :]  # All but last position
shift_labels = input_ids[:, 1:]    # All but first position

# Compute loss
loss = F.cross_entropy(
    shift_logits.reshape(-1, vocab_size),
    shift_labels.reshape(-1)
)

# This measures: "How well can model predict next token?"
```

### 2.4 Optimizers

**What do they do?**
Update model weights to minimize loss.

**Gradient Descent (Basic Idea):**
```
weight_new = weight_old - learning_rate * gradient
```

**Adam Optimizer (What We Use):**

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,           # Learning rate
    betas=(0.9, 0.95), # Momentum parameters
    eps=1e-8,          # Numerical stability
    weight_decay=0.1   # L2 regularization
)

# Training loop
for batch in data_loader:
    # Forward pass
    output = model(batch)
    loss = compute_loss(output, target)
    
    # Backward pass
    loss.backward()  # Compute gradients
    
    # Update weights
    optimizer.step()
    
    # Clear gradients
    optimizer.zero_grad()
```

**Why Adam?**
- Adapts learning rate per parameter
- Uses momentum (smooth updates)
- Works well in practice
- Industry standard

---

## Part 3: Transformer Architecture Theory

### 3.1 The Big Picture

**What Problem Do Transformers Solve?**
Understanding sequences (text, audio, video) where context matters.

**Example:**
- "The bank is near the river" - bank = riverbank
- "I need to bank the check" - bank = financial institution

Transformers use **attention** to understand context.

### 3.2 Attention Mechanism

**Core Idea:**
When processing a word, look at ALL other words to understand context.

**Mathematical Formula:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

Where:
- Q (Query): "What am I looking for?"
- K (Key): "What do I contain?"
- V (Value): "What information do I have?"
```

**Intuitive Example:**

```python
# Sentence: "The cat sat on the mat"
# When processing "sat":

# Query: "What's happening?"
# Keys from all words:
#   "The" -> not relevant
#   "cat" -> very relevant (who is sitting?)
#   "sat" -> somewhat relevant (the action itself)
#   "on" -> relevant (where?)
#   "the" -> not relevant
#   "mat" -> very relevant (where sitting?)

# Attention weights (after softmax):
weights = [0.02, 0.45, 0.15, 0.10, 0.02, 0.26]

# Multiply each word's value by its weight
# Result: weighted combination emphasizing "cat" and "mat"
```

**Code Implementation:**

```python
def attention(q, k, v, mask=None):
    # q, k, v shapes: [batch, heads, seq_len, head_dim]
    
    d_k = q.size(-1)
    
    # Compute attention scores
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    # Shape: [batch, heads, seq_len, seq_len]
    
    # Apply mask (prevent looking at future tokens)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # Convert to probabilities
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum of values
    output = torch.matmul(attn_weights, v)
    
    return output, attn_weights
```

### 3.3 Multi-Head Attention

**Why Multiple Heads?**
Different heads can focus on different aspects:
- Head 1: Subject-verb relationships
- Head 2: Adjective-noun relationships
- Head 3: Long-range dependencies
- etc.

**Visualization:**

```
Input: "The quick brown fox"

Head 1 (syntax):
  The → quick (0.8)  # Adjective follows article
  quick → brown (0.7)
  brown → fox (0.9)

Head 2 (semantics):
  fox → quick (0.6)  # Fast animal
  fox → brown (0.8)  # Color of animal

Head 3 (long-range):
  All words attend to "fox" (the main subject)
```

**Code:**

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Project to Q, K, V
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        batch_size, seq_len, hidden_size = x.shape
        
        # Project and split into heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention for each head
        attn_output = attention(q, k, v)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, hidden_size)
        
        # Final projection
        output = self.o_proj(attn_output)
        
        return output
```

### 3.4 Grouped Query Attention (GQA)

**Problem with Standard Attention:**
Each head has its own K and V matrices → memory expensive

**GQA Solution:**
Multiple heads share K and V matrices

```python
# Standard Multi-Head Attention
# 12 heads, 12 separate K matrices, 12 separate V matrices

# Grouped Query Attention
# 12 query heads, 4 KV heads
# Heads 0-2 share KV head 0
# Heads 3-5 share KV head 1
# Heads 6-8 share KV head 2
# Heads 9-11 share KV head 3

class GroupedQueryAttention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        # Full Q projection
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        
        # Smaller K, V projections
        kv_size = num_kv_heads * self.head_dim
        self.k_proj = nn.Linear(hidden_size, kv_size)
        self.v_proj = nn.Linear(hidden_size, kv_size)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        # ... project Q, K, V ...
        
        # Expand K and V to match Q heads
        k = k.repeat_interleave(self.num_kv_groups, dim=1)
        v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Now K, V match Q dimensions
        # ... compute attention ...
```

**Benefits:**
- 4x less memory for KV cache
- Minimal quality loss
- Much faster inference

### 3.5 Position Embeddings

**Problem:**
Attention has no notion of word order!
- "Dog bites man" vs "Man bites dog" would be identical

**Solution: Add Position Information**

**Rotary Position Embedding (RoPE) - What We Use:**

Instead of adding position vectors, RoPE rotates the query and key vectors based on position.

```python
def apply_rotary_pos_emb(q, k, cos, sin):
    """
    Rotate vectors based on position
    
    Position 0: No rotation
    Position 1: Small rotation
    Position 2: More rotation
    etc.
    
    Effect: Nearby positions have similar rotations
    → Naturally encodes "distance" between tokens
    """
    # Split into two halves
    q1, q2 = q[..., :q.shape[-1]//2], q[..., q.shape[-1]//2:]
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
    
    # Apply rotation
    q_rotated = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos
    ], dim=-1)
    
    k_rotated = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos
    ], dim=-1)
    
    return q_rotated, k_rotated
```

**Why RoPE is Better:**
- Generalizes to longer sequences
- No learned parameters
- Works better empirically

### 3.6 Feed-Forward Network (FFN)

**Purpose:**
Process each token individually after attention gathered context.

**Standard FFN:**
```python
class FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)          # Expand
        x = self.activation(x)   # Non-linearity
        x = self.fc2(x)          # Compress
        return x
```

**SwiGLU (What We Use) - Better Performance:**

```python
class SwiGLU(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
    
    def forward(self, x):
        # Two parallel projections
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        
        # Element-wise multiplication (gating)
        gated = gate * up
        
        # Project back
        output = self.down_proj(gated)
        return output
```

**Why Gating Works:**
The gate controls information flow:
- Gate ≈ 0: Block this information
- Gate ≈ 1: Pass this information
- Learned dynamically during training

### 3.7 Layer Normalization

**Problem:**
As signals pass through layers, they can explode or vanish.

**Layer Norm Solution:**
Normalize activations to have mean=0, std=1

**Standard LayerNorm:**
```python
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
    
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
```

**RMSNorm (What We Use) - Simpler and Faster:**

```python
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    
    def forward(self, x):
        # Root Mean Square normalization
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x * rms
        return self.weight * x_norm
```

**Benefits:**
- No bias parameter
- No mean subtraction (faster)
- Empirically works as well or better

### 3.8 Complete Transformer Block

**Putting It All Together:**

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # Attention
        self.attention = GroupedQueryAttention(config)
        self.attention_norm = RMSNorm(config.hidden_size)
        
        # Feed-forward
        self.feed_forward = SwiGLU(config)
        self.ffn_norm = RMSNorm(config.hidden_size)
    
    def forward(self, x):
        # Pre-norm + Attention + Residual
        residual = x
        x = self.attention_norm(x)
        x = self.attention(x)
        x = residual + x  # Residual connection
        
        # Pre-norm + FFN + Residual
        residual = x
        x = self.ffn_norm(x)
        x = self.feed_forward(x)
        x = residual + x  # Residual connection
        
        return x
```

**Key Design Choices:**

1. **Pre-normalization** (norm before layer):
   - More stable training
   - Better gradient flow

2. **Residual connections** (x = residual + layer(x)):
   - Allows gradients to flow directly
   - Enables training very deep networks
   - Model can learn identity mapping if needed

3. **Order matters:**
   - Attention first (gather context)
   - FFN second (process individually)

---

## Part 4: Training Process Deep Dive

### 4.1 Forward Pass

**What Happens:**
Input data flows through network to produce output.

```python
def forward_pass_explained():
    # Input: Token IDs
    input_ids = torch.tensor([[15, 496, 995, 67]])  # [1, 4]
    
    # Step 1: Embedding
    # Convert token IDs to dense vectors
    embedded = embedding(input_ids)  # [1, 4, 768]
    # Each token is now a 768-dimensional vector
    
    # Step 2: Add position information
    embedded_with_pos = add_rotary_embeddings(embedded)
    
    # Step 3: Pass through transformer layers
    hidden_states = embedded_with_pos
    for layer in transformer_layers:
        # Each layer:
        # - Applies attention (looks at context)
        # - Applies FFN (processes information)
        # - Uses residual connections
        hidden_states = layer(hidden_states)  # [1, 4, 768]
    
    # Step 4: Final normalization
    normalized = layer_norm(hidden_states)  # [1, 4, 768]
    
    # Step 5: Project to vocabulary
    logits = output_projection(normalized)  # [1, 4, 32000]
    # For each position, probability distribution over all tokens
    
    return logits
```

**Understanding Logits:**

```python
# Logits are raw scores (before softmax)
logits = torch.tensor([[2.5, 1.0, 0.3, ...]])  # 32000 values

# Convert to probabilities
probs = F.softmax(logits, dim=-1)
# Now sums to 1.0

# Most likely next token
next_token = logits.argmax()
```

### 4.2 Loss Computation

**Next-Token Prediction:**

```python
def compute_loss_explained():
    # Input sequence
    input_ids = [15, 496, 995, 67, 89]
    
    # Forward pass
    logits = model(input_ids)  # [1, 5, 32000]
    
    # For next-token prediction:
    # - Use tokens 0-3 to predict token 1-4
    # - Token 4 has no label (nothing after it)
    
    # Shift logits and labels
    shift_logits = logits[:, :-1, :]  # [1, 4, 32000] - predictions
    shift_labels = input_ids[:, 1:]    # [1, 4] - actual next tokens
    
    # Compute cross-entropy loss
    loss = F.cross_entropy(
        shift_logits.reshape(-1, 32000),  # [4, 32000]
        shift_labels.reshape(-1)           # [4]
    )
    
    # Loss measures: "How confident was model in correct token?"
    # Low loss = high confidence in correct tokens
    # High loss = low confidence or wrong predictions
    
    return loss
```

**Why Shift?**

```
Input:     [The,  cat,  sat,  on,   the]
Predict:        [cat,  sat,  on,   the,  mat]

Position 0 predicts position 1: "The" → "cat"
Position 1 predicts position 2: "cat" → "sat"
Position 2 predicts position 3: "sat" → "on"
Position 3 predicts position 4: "on" → "the"
```

### 4.3 Backward Pass (Backpropagation)

**What Happens:**
Compute how much each parameter contributed to the loss.

```python
def backward_pass_explained():
    # After forward pass, we have loss
    loss = compute_loss(...)  # Scalar value
    
    # Backward pass computes gradients
    loss.backward()
    
    # This calculates ∂loss/∂param for EVERY parameter
    # Using chain rule automatically
    
    # Example gradient flow:
    # loss
    #   ↓ ∂loss/∂logits
    # logits
    #   ↓ ∂loss/∂hidden
    # hidden_states
    #   ↓ ∂loss/∂attention_output
    # attention_output
    #   ↓ ∂loss/∂Q, ∂loss/∂K, ∂loss/∂V
    # Q, K, V
    #   ↓ ∂loss/∂weights
    # weight matrices
    
    # After backward(), all parameters have .grad attribute
    for name, param in model.named_parameters():
        print(f"{name}: gradient shape {param.grad.shape}")
```

**Chain Rule Example:**

```
Suppose: z = f(g(x))

Then: ∂z/∂x = ∂z/∂g × ∂g/∂x

PyTorch does this automatically for complex networks!

Example:
x → Linear → ReLU → Linear → Loss

∂Loss/∂w1 = ∂Loss/∂Linear2 × ∂Linear2/∂ReLU × ∂ReLU/∂Linear1 × ∂Linear1/∂w1
```

### 4.4 Optimizer Step

**Gradient Descent:**

```python
# Basic gradient descent
for param in model.parameters():
    param.data = param.data - learning_rate * param.grad
```

**Adam Optimizer (What We Use):**

```python
class AdamOptimizer:
    """
    Adam: Adaptive Moment Estimation
    Combines momentum and adaptive learning rates
    """
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999)):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        
        # Running averages
        self.m = [torch.zeros_like(p) for p in params]  # First moment
        self.v = [torch.zeros_like(p) for p in params]  # Second moment
        self.t = 0  # Timestep
    
    def step(self):
        self.t += 1
        
        for i, param in enumerate(self.params):
            grad = param.grad
            
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            
            # Update biased second raw moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * grad**2
            
            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)
            
            # Update parameters
            param.data = param.data - self.lr * m_hat / (torch.sqrt(v_hat) + 1e-8)
```

**Why Adam Works Well:**

1. **Momentum** (m): Smooths updates, accelerates convergence
2. **Adaptive learning rates** (v): Different rate per parameter
3. **Bias correction**: Accounts for initialization at zero

### 4.5 Complete Training Loop

```python
def training_loop_explained():
    model = LLMModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    for epoch in range(num_epochs):
        for batch in dataloader:
            # 1. Get data
            input_ids = batch['input_ids'].to(device)
            
            # 2. Forward pass
            logits, loss = model(input_ids, labels=input_ids)
            
            # 3. Backward pass
            loss.backward()  # Compute gradients
            
            # 4. Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 5. Optimizer step
            optimizer.step()  # Update weights
            
            # 6. Clear gradients
            optimizer.zero_grad()  # Important! Otherwise gradients accumulate
            
            # 7. Log progress
            print(f"Loss: {loss.item():.4f}")
```

### 4.6 Learning Rate Scheduling

**Why Schedule Learning Rate?**
- Start: Large LR for fast progress
- Middle: Moderate LR for steady improvement  
- End: Small LR for fine-tuning

**Cosine Schedule (What We Use):**

```python
def get_cosine_schedule(step, warmup_steps, max_steps, max_lr, min_lr):
    # Warmup phase: Linear increase
    if step < warmup_steps:
        return max_lr * step / warmup_steps
    
    # Cosine decay phase
    if step > max_steps:
        return min_lr
    
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine_decay

# Visualized:
#     max_lr ────╱────────╲
#              ╱            ╲
#            ╱                ╲___
#          ╱                      ────── min_lr
#        0     warmup    training    max_steps
```

**In Our Code:**

```python
def train_step(self):
    # Update learning rate
    lr = get_lr(self.step, self.config)
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
    
    # ... rest of training step ...
```

### 4.7 Gradient Accumulation

**Problem:**
Limited GPU memory → can't fit large batch size

**Solution:**
Accumulate gradients over multiple small batches

```python
def gradient_accumulation_explained():
    micro_batch_size = 8   # What fits in memory
    batch_size = 32        # Desired effective batch size
    accumulation_steps = batch_size // micro_batch_size  # 4
    
    optimizer.zero_grad()
    
    for i, micro_batch in enumerate(dataloader):
        # Forward pass
        output = model(micro_batch)
        loss = compute_loss(output)
        
        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        
        # Backward pass (accumulates gradients)
        loss.backward()
        
        # Update weights every N steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

# Effect: Same as training with batch_size=32
# But uses only memory for batch_size=8
```

### 4.8 Mixed Precision Training

**Idea:**
Use FP16 (16-bit) instead of FP32 (32-bit) for speed and memory savings

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()  # Handles gradient scaling

for batch in dataloader:
    optimizer.zero_grad()
    
    # Forward pass in FP16
    with autocast():
        output = model(batch)
        loss = compute_loss(output)
    
    # Backward pass with scaled loss
    scaler.scale(loss).backward()
    
    # Unscale gradients and update
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    scaler.step(optimizer)
    scaler.update()
```

**Why Scale Gradients?**
FP16 has smaller range than FP32 → gradients can underflow to zero
Solution: Scale up during backward, scale down before optimizer step

**Benefits:**
- 2x faster training
- 2x less memory
- Same final quality (with proper scaling)

---

## Part 5: Distributed Training Concepts

### 5.1 Data Parallel Training

**Idea:**
Split batch across multiple GPUs

```
Batch: 128 examples

GPU 0: Examples 0-31   ──┐
GPU 1: Examples 32-63  ──┤
GPU 2: Examples 64-95  ──┼─→ All compute gradients
GPU 3: Examples 96-127 ──┘

Then: Average gradients across all GPUs
Finally: All GPUs update with same averaged gradient
```

**Code:**

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Create model and move to GPU
model = LLMModel(config).to(f'cuda:{rank}')

# Wrap with DDP
model = DDP(model, device_ids=[rank])

# Training loop
for batch in dataloader:
    # Each GPU processes different part of batch
    output = model(batch)
    loss = compute_loss(output)
    
    # Backward pass
    loss.backward()
    
    # DDP automatically:
    # 1. Averages gradients across all GPUs
    # 2. Ensures all GPUs have same gradients
    
    optimizer.step()
    optimizer.zero_grad()
```

### 5.2 Distributed Data Sampler

**Problem:**
Each GPU needs different data

**Solution:**
Use DistributedSampler

```python
from torch.utils.data import DistributedSampler

# Create sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,  # Total number of GPUs
    rank=rank,                 # Current GPU ID
    shuffle=True
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=32,
    sampler=sampler,  # Ensures each GPU gets different data
    num_workers=4
)

# In training loop
for epoch in range(num_epochs):
    sampler.set_epoch(epoch)  # Important for shuffling
    
    for batch in dataloader:
        # Train as usual
        ...
```

### 5.3 Gradient Synchronization

**What Happens in DDP:**

```python
# Simplified version of what DDP does

class SimpleDDP:
    def backward(self, loss):
        # Normal backward pass
        loss.backward()
        
        # After backward, synchronize gradients
        for param in self.parameters():
            # AllReduce: Sum gradients from all GPUs
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            
            # Average
            param.grad /= world_size
        
        # Now all GPUs have identical gradients
```

**Communication Patterns:**

```
All-Reduce Operation:

GPU 0: grad = [1, 2, 3]
GPU 1: grad = [4, 5, 6]
GPU 2: grad = [7, 8, 9]

After all-reduce:
GPU 0: grad = [12, 15, 18]  # Sum
GPU 1: grad = [12, 15, 18]
GPU 2: grad = [12, 15, 18]

After averaging:
All GPUs: grad = [4, 5, 6]  # Average
```

### 5.4 Multi-Node Training

**Setup:**

```
Node 0 (Master):
  - GPU 0, 1, 2, 3
  - IP: 192.168.1.10
  - Rank: 0-3

Node 1 (Worker):
  - GPU 0, 1, 2, 3
  - IP: 192.168.1.11
  - Rank: 4-7

Total: 8 GPUs across 2 nodes
```

**Initialization:**

```python
import os

# Environment variables set by torchrun
rank = int(os.environ['RANK'])              # 0-7
world_size = int(os.environ['WORLD_SIZE'])  # 8
local_rank = int(os.environ['LOCAL_RANK'])  # 0-3 on each node

# Initialize process group
dist.init_process_group(
    backend='nccl',  # NVIDIA GPUs
    init_method=f'tcp://{master_addr}:{master_port}',
    rank=rank,
    world_size=world_size
)

# Rest is same as single-node!
```

### 5.5 Communication Overhead

**Understanding Bottlenecks:**

```python
# Training step breakdown:

# 1. Forward pass - No communication needed
forward_time = compute_forward()

# 2. Backward pass - Gradients synchronized
backward_time = compute_backward() + communication_time

# Communication time depends on:
# - Model size (more parameters = more data to sync)
# - Network bandwidth (InfiniBand > Ethernet)
# - Number of GPUs (more GPUs = more communication)

# Typical split:
# - Forward: 40% of time
# - Backward (compute): 40% of time
# - Communication: 20% of time

# With fast network: ~10% communication overhead
# With slow network: ~50% communication overhead
```

**Optimization:**

```python
# DDP automatically overlaps computation and communication
# While computing gradients for layer N, 
# simultaneously transmit gradients for layer N-1

# This is why DDP is much faster than naive data parallel
```

---

## Part 6: Code Walkthrough with Theory

### 6.1 Embedding Layer Deep Dive

**From model.py:**

```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```

**What It Does:**

```python
# Embedding is essentially a lookup table

vocab_size = 32000
hidden_size = 768

# Embedding matrix shape: [32000, 768]
# Each row corresponds to one token

embedding_matrix = torch.randn(vocab_size, hidden_size)

# Input: token IDs
input_ids = torch.tensor([15, 496, 995])

# Output: Look up corresponding rows
embedded = embedding_matrix[input_ids]
# Shape: [3, 768]

# Token 15 → embedding_matrix[15] (768-dimensional vector)
# Token 496 → embedding_matrix[496]
# Token 995 → embedding_matrix[995]
```

**Why Embeddings Work:**

During training, similar words get similar embeddings:
- "cat" → [0.2, -0.5, 0.8, ...]
- "dog" → [0.3, -0.4, 0.7, ...]  # Similar to cat
- "car" → [0.9, 0.2, -0.3, ...]  # Different from cat/dog

### 6.2 Attention Mechanism in Code

**From model.py:**

```python
class GroupedQueryAttention(nn.Module):
    def forward(self, hidden_states):
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head
        q = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention computation
        attn_output = F.scaled_dot_product_attention(q, k, v)
        
        return attn_output
```

**Step-by-Step Walkthrough:**

```python
# Input shape: [batch=2, seq_len=10, hidden=768]

# 1. Projection
q = self.q_proj(hidden_states)  # [2, 10, 768]
k = self.k_proj(hidden_states)  # [2, 10, 768]
v = self.v_proj(hidden_states)  # [2, 10, 768]

# 2. Reshape for multi-head (num_heads=12)
# Split 768 dimensions into 12 heads of 64 dimensions each
q = q.view(2, 10, 12, 64)       # [batch, seq, heads, head_dim]
q = q.transpose(1, 2)            # [batch, heads, seq, head_dim]
# Final shape: [2, 12, 10, 64]

# 3. Apply rotary embeddings (position encoding)
q, k = apply_rotary_pos_emb(q, k, cos, sin)
# Rotates vectors based on position

# 4. Compute attention scores
scores = torch.matmul(q, k.transpose(-2, -1))  # [2, 12, 10, 10]
# scores[i, j, p, q] = similarity between position p and q in head j of batch i

scores = scores / math.sqrt(64)  # Scale by sqrt(head_dim)

# 5. Apply softmax (convert to probabilities)
attn_weights = F.softmax(scores, dim=-1)  # [2, 12, 10, 10]
# Each row sums to 1

# 6. Weighted sum of values
attn_output = torch.matmul(attn_weights, v)  # [2, 12, 10, 64]

# 7. Concatenate heads
attn_output = attn_output.transpose(1, 2)  # [2, 10, 12, 64]
attn_output = attn_output.contiguous().view(2, 10, 768)

# 8. Final projection
output = self.o_proj(attn_output)  # [2, 10, 768]
```

### 6.3 Training Step Explained

**From train.py:**

```python
def train_step(self, batch):
    input_ids = batch['input_ids'].to(self.device)
    
    # Forward pass
    with autocast(enabled=(self.scaler is not None)):
        logits, loss = self.model(input_ids, labels=input_ids)
        loss = loss / self.config.gradient_accumulation_steps
    
    # Backward pass
    if self.scaler is not None:
        self.scaler.scale(loss).backward()
    else:
        loss.backward()
    
    return loss.item()
```

**Detailed Explanation:**

```python
# Example batch
input_ids = torch.tensor([
    [15, 496, 995, 67],
    [89, 12, 334, 90]
])  # Shape: [2, 4]

# 1. Move to GPU
input_ids = input_ids.to('cuda')

# 2. Forward pass with mixed precision
with autocast():
    # Model converts to FP16 internally for speed
    
    # a. Embedding
    embedded = self.embed_tokens(input_ids)  # [2, 4, 768]
    
    # b. Transformer layers
    for layer in self.layers:
        embedded = layer(embedded)
    
    # c. Output projection
    logits = self.lm_head(embedded)  # [2, 4, 32000]
    
    # d. Compute loss
    shift_logits = logits[:, :-1, :]   # [2, 3, 32000]
    shift_labels = input_ids[:, 1:]     # [2, 3]
    
    loss = F.cross_entropy(
        shift_logits.reshape(-1, 32000),  # [6, 32000]
        shift_labels.reshape(-1)           # [6]
    )

# 3. Scale loss for gradient accumulation
# If accumulating over 4 steps, divide by 4
loss = loss / 4

# 4. Backward pass with gradient scaling
scaler.scale(loss).backward()
# Scales loss by a large number to prevent FP16 underflow
# Then computes gradients

# 5. Gradient clipping (after accumulation is done)
if accumulation_complete:
    scaler.unscale_(optimizer)  # Unscale gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # 6. Optimizer step
    scaler.step(optimizer)
    scaler.update()
    
    # 7. Clear gradients
    optimizer.zero_grad()
```

### 6.4 Checkpoint Saving/Loading

**From train.py:**

```python
def save_checkpoint(self):
    checkpoint = {
        'step': self.step,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'config': asdict(self.config),
    }
    torch.save(checkpoint, save_path)
```

**Understanding state_dict:**

```python
# state_dict() returns dictionary of all parameters

model_state = model.state_dict()
# {
#     'embed_tokens.weight': tensor([...]),  # [32000, 768]
#     'layers.0.attention.q_proj.weight': tensor([...]),
#     'layers.0.attention.k_proj.weight': tensor([...]),
#     ...
#     'lm_head.weight': tensor([...]),  # [32000, 768]
# }

# Save to disk
torch.save(model_state, 'model.pt')

# Load from disk
model_state = torch.load('model.pt')
model.load_state_dict(model_state)

# Now model has exact same weights
```

**Optimizer State:**

```python
# Optimizer also has state (momentum, etc.)
optimizer_state = optimizer.state_dict()
# {
#     'state': {
#         0: {'step': 1000, 'exp_avg': tensor([...]), 'exp_avg_sq': tensor([...])},
#         1: {'step': 1000, 'exp_avg': tensor([...]), 'exp_avg_sq': tensor([...])},
#         ...
#     },
#     'param_groups': [...]
# }

# Load to resume training with same momentum
optimizer.load_state_dict(optimizer_state)
```

---

## Part 7: Mathematical Foundations

### 7.1 Linear Algebra Basics

**Vectors:**
```python
# Vector: List of numbers
v = torch.tensor([1, 2, 3])

# Operations:
v + v    # [2, 4, 6] - element-wise addition
v * 2    # [2, 4, 6] - scalar multiplication
v * v    # [1, 4, 9] - element-wise multiplication (Hadamard)
v @ v    # 14 - dot product (1*1 + 2*2 + 3*3)
```

**Matrices:**
```python
# Matrix: 2D array
A = torch.tensor([[1, 2], [3, 4], [5, 6]])  # 3x2

# Matrix-vector multiplication
v = torch.tensor([1, 2])
result = A @ v  # [5, 11, 17]

# How:
# [1*1 + 2*2]   [5]
# [3*1 + 4*2] = [11]
# [5*1 + 6*2]   [17]

# Matrix-matrix multiplication
B = torch.tensor([[1, 2, 3], [4, 5, 6]])  # 2x3
C = A @ B  # 3x3

# Rule: (m x n) @ (n x p) = (m x p)
```

**In Neural Networks:**

```python
# Linear layer: y = Wx + b

W = torch.randn(5, 10)  # Weight matrix
x = torch.randn(32, 10)  # Batch of 32, each with 10 features
b = torch.randn(5)       # Bias

y = x @ W.T + b  # Result: [32, 5]

# This is what nn.Linear does!
```

### 7.2 Calculus - Derivatives and Gradients

**Derivative Basics:**

```
f(x) = x²
f'(x) = 2x

Meaning: "How much does f change when x changes?"

At x=3:
f'(3) = 6
→ If x increases by 0.1, f increases by ~0.6
```

**Gradients (multi-variable):**

```
f(x, y) = x² + y²
∇f = [∂f/∂x, ∂f/∂y] = [2x, 2y]

At (3, 4):
∇f = [6, 8]
→ Moving in direction [6, 8] increases f fastest
→ Moving in direction [-6, -8] decreases f fastest
```

**In Neural Networks:**

```python
# Loss is function of all parameters
loss = f(w1, w2, w3, ..., wn)

# Gradient tells us how to change each weight
∇loss = [∂loss/∂w1, ∂loss/∂w2, ...]

# Update rule:
w_new = w_old - learning_rate * ∇loss

# This is gradient descent!
```

### 7.3 Chain Rule

**Single Variable:**

```
If z = f(g(x)), then:
dz/dx = df/dg × dg/dx

Example:
z = (x + 2)²
Let g(x) = x + 2, then z = g²

dz/dx = 2g × 1 = 2(x + 2)
```

**Neural Networks:**

```
Input → Layer1 → Layer2 → Loss

∂Loss/∂w1 = ∂Loss/∂Layer2 × ∂Layer2/∂Layer1 × ∂Layer1/∂w1

PyTorch computes this automatically with .backward()!
```

**Example:**

```python
x = torch.tensor([2.0], requires_grad=True)
y = x + 2        # y = x + 2
z = y ** 2       # z = y²

z.backward()     # Computes dz/dx

print(x.grad)    # 2(x+2) = 2(2+2) = 8
```

### 7.4 Softmax Function

**Formula:**
```
softmax(x_i) = e^(x_i) / Σ(e^(x_j))
```

**Purpose:**
Convert scores to probabilities (sum to 1)

```python
logits = torch.tensor([2.0, 1.0, 0.1])

# Raw softmax
exp_logits = torch.exp(logits)
# [7.39, 2.72, 1.11]

probs = exp_logits / exp_logits.sum()
# [0.66, 0.24, 0.10]

# Properties:
# 1. All positive
# 2. Sum to 1.0
# 3. Highest logit → highest probability
```

**In Practice:**

```python
# For numerical stability, subtract max
logits = torch.tensor([1000, 999, 998])  # Would overflow!

logits_stable = logits - logits.max()  # [-2, -1, 0]
probs = F.softmax(logits_stable, dim=0)
# Same result, no overflow
```

### 7.5 Cross-Entropy Loss

**Formula:**
```
CrossEntropy = -log(p_correct)

Where p_correct is probability assigned to correct class
```

**Example:**

```python
# Model outputs probabilities
probs = torch.tensor([0.7, 0.2, 0.1])  # 3 classes

# True label is class 0
# CrossEntropy = -log(0.7) = 0.357

# True label is class 2
# CrossEntropy = -log(0.1) = 2.303  # Much higher loss!

# Intuition:
# High probability on correct class → Low loss
# Low probability on correct class → High loss
```

**Why Log?**

```
Probability vs Loss:
P = 1.0  → Loss = 0.0   (perfect)
P = 0.5  → Loss = 0.69
P = 0.1  → Loss = 2.30
P = 0.01 → Loss = 4.60  (very bad)

Log heavily penalizes low probabilities
```

**In Code:**

```python
# Logits (before softmax)
logits = torch.tensor([[2.0, 1.0, 0.1]])  # [1, 3]
target = torch.tensor([0])  # Correct class is 0

# Cross-entropy does softmax internally
loss = F.cross_entropy(logits, target)

# Equivalent to:
probs = F.softmax(logits, dim=1)
loss = -torch.log(probs[0, target[0]])
```

---

## Summary: Connecting Theory to Code

### Key Takeaways

1. **PyTorch Tensors** = Multi-dimensional arrays that can run on GPU
2. **Autograd** = Automatic differentiation (computes gradients)
3. **nn.Module** = Building block for neural networks
4. **Forward Pass** = Data flows through network to produce output
5. **Loss** = Measure of how wrong predictions are
6. **Backward Pass** = Compute gradients using chain rule
7. **Optimizer** = Update weights using gradients
8. **Attention** = Mechanism to focus on relevant parts of input
9. **Transformers** = Stack of attention + feed-forward layers
10. **Distributed Training** = Split work across multiple GPUs

### Mapping Theory to Our Code

| Concept | File | What It Does |
|---------|------|--------------|
| Embeddings | model.py | Convert token IDs to vectors |
| Attention | model.py | Focus on relevant context |
| Feed-Forward | model.py | Process information |
| Transformer Block | model.py | Attention + FFN with residuals |
| Loss Computation | train.py | Measure prediction error |
| Backpropagation | train.py | Compute gradients (automatic) |
| Optimizer | train.py | Update weights using gradients |
| Data Loading | train.py | Batch and shuffle data |
| Distributed Training | train.py | Coordinate multiple GPUs |
| Checkpointing | train.py | Save/load model state |

---

## Practical Exercises

### Exercise 1: Understanding Tensors

```python
import torch

# Create a simple tensor
x = torch.tensor([[1, 2, 3], [4, 5, 6]])

# Questions:
# 1. What is the shape? 
print(x.shape)  # Answer: torch.Size([2, 3])

# 2. What is the total number of elements?
print(x.numel())  # Answer: 6

# 3. Reshape to [3, 2]
y = x.view(3, 2)
print(y)

# 4. Add 10 to all elements
z = x + 10
print(z)

# 5. Compute mean
mean = x.float().mean()
print(mean)  # Answer: 3.5
```

### Exercise 2: Simple Forward Pass

```python
import torch
import torch.nn as nn

# Create a tiny model
class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(5, 10)
        self.layer2 = nn.Linear(10, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

model = TinyModel()

# Create input
x = torch.randn(3, 5)  # Batch of 3, 5 features

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")  # [3, 2]

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
# layer1: 5*10 + 10 = 60
# layer2: 10*2 + 2 = 22
# Total: 82
```

### Exercise 3: Compute Loss and Gradients

```python
import torch
import torch.nn.functional as F

# Model from previous exercise
model = TinyModel()

# Input and target
x = torch.randn(3, 5)
target = torch.tensor([0, 1, 0])  # Class labels

# Forward pass
output = model(x)

# Compute loss
loss = F.cross_entropy(output, target)
print(f"Loss: {loss.item():.4f}")

# Backward pass
loss.backward()

# Check gradients
for name, param in model.named_parameters():
    print(f"{name}: gradient shape {param.grad.shape}")
    print(f"  Gradient mean: {param.grad.mean().item():.6f}")
```

### Exercise 4: Training Loop

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Create model, optimizer
model = TinyModel()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Fake dataset
def create_batch():
    x = torch.randn(32, 5)
    y = torch.randint(0, 2, (32,))
    return x, y

# Training loop
for step in range(100):
    # Get batch
    x, y = create_batch()
    
    # Forward pass
    output = model(x)
    loss = F.cross_entropy(output, y)
    
    # Backward pass
    loss.backward()
    
    # Update weights
    optimizer.step()
    optimizer.zero_grad()
    
    # Log
    if step % 10 == 0:
        print(f"Step {step}: Loss = {loss.item():.4f}")

# After training, test accuracy
x_test, y_test = create_batch()
with torch.no_grad():
    output = model(x_test)
    pred = output.argmax(dim=1)
    accuracy = (pred == y_test).float().mean()
    print(f"Test accuracy: {accuracy.item():.2%}")
```

### Exercise 5: Understanding Attention

```python
import torch
import torch.nn.functional as F

# Simplified attention mechanism
def simple_attention(query, key, value):
    """
    Args:
        query: [batch, seq_len, dim]
        key: [batch, seq_len, dim]
        value: [batch, seq_len, dim]
    """
    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1))
    scores = scores / (query.size(-1) ** 0.5)
    
    # Convert to probabilities
    attn_weights = F.softmax(scores, dim=-1)
    
    # Weighted sum
    output = torch.matmul(attn_weights, value)
    
    return output, attn_weights

# Example
batch = 1
seq_len = 4
dim = 8

q = torch.randn(batch, seq_len, dim)
k = torch.randn(batch, seq_len, dim)
v = torch.randn(batch, seq_len, dim)

output, weights = simple_attention(q, k, v)

print("Attention weights:")
print(weights[0])  # [4, 4] matrix
# Each row: how much each position attends to others
# Each row sums to 1.0
```

---

## Advanced Topics

### Topic 1: Gradient Checkpointing

**Problem:** Large models run out of memory

**Solution:** Trade computation for memory

```python
from torch.utils.checkpoint import checkpoint

class TransformerWithCheckpointing(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # Use checkpointing: recompute forward pass during backward
            x = checkpoint(layer, x)
        return x

# Benefits:
# - 50% less memory usage
# - 20% slower training (due to recomputation)
# - Allows training larger models
```

### Topic 2: Flash Attention

**Standard Attention Problem:**
- Materializes full attention matrix: [seq_len, seq_len]
- For seq_len=2048: 4M values!
- Most are never used (softmax makes many near-zero)

**Flash Attention Solution:**
- Compute attention in blocks
- Never materialize full matrix
- Fuse operations to reduce memory reads/writes

```python
# Standard attention
def standard_attention(q, k, v):
    # Memory: O(seq_len²)
    attn = (q @ k.T) / sqrt(d)      # Materialize full matrix
    attn = softmax(attn)
    output = attn @ v
    return output

# Flash attention (simplified concept)
def flash_attention(q, k, v):
    # Memory: O(seq_len)
    # Process in blocks, never store full attention matrix
    # Implemented in CUDA for maximum speed
    return F.scaled_dot_product_attention(q, k, v)

# In our code:
if self.use_flash:
    attn_output = F.scaled_dot_product_attention(q, k, v)
# Automatically uses Flash Attention if available!
```

### Topic 3: Quantization

**Idea:** Use fewer bits per parameter

```python
import torch
from torch.quantization import quantize_dynamic

# Original model (FP32)
model = LLMModel(config)  # 4 bytes per parameter

# Quantize to INT8
quantized_model = quantize_dynamic(
    model,
    {nn.Linear},  # Which layers to quantize
    dtype=torch.qint8  # 1 byte per parameter
)

# Result:
# - 4x smaller model
# - Faster inference
# - Slight accuracy loss (~1-2%)

# Save space
torch.save(quantized_model.state_dict(), 'model_int8.pt')
```

### Topic 4: Model Parallelism

**When Data Parallelism Isn't Enough:**
Model is too large to fit on single GPU

**Solution: Split Model Across GPUs**

```python
class ModelParallelLLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # First half of layers on GPU 0
        self.layers_0 = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers // 2)
        ]).to('cuda:0')
        
        # Second half on GPU 1
        self.layers_1 = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers // 2)
        ]).to('cuda:1')
    
    def forward(self, x):
        # Start on GPU 0
        x = x.to('cuda:0')
        for layer in self.layers_0:
            x = layer(x)
        
        # Move to GPU 1
        x = x.to('cuda:1')
        for layer in self.layers_1:
            x = layer(x)
        
        return x

# Note: Simpler alternatives exist (DeepSpeed, Megatron-LM)
```

### Topic 5: Learning Rate Warmup

**Why Warmup?**
Large learning rates at start can destabilize training

**Solution:** Gradually increase LR

```python
def warmup_then_cosine_decay(step, warmup_steps, max_steps, max_lr, min_lr):
    if step < warmup_steps:
        # Linear warmup
        return max_lr * (step / warmup_steps)
    else:
        # Cosine decay
        progress = (step - warmup_steps) / (max_steps - warmup_steps)
        return min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

# Example trajectory:
# Step 0:    LR = 0
# Step 500:  LR = 1.5e-4
# Step 1000: LR = 3e-4  (max)
# Step 5000: LR = 2.5e-4
# Step 10000: LR = 2e-4
# Step 50000: LR = 5e-5
# Step 100000: LR = 3e-5 (min)
```

---

## Debugging Tips

### Tip 1: Check Tensor Shapes

```python
# Add shape assertions
def forward(self, x):
    # x should be [batch, seq_len, hidden]
    assert len(x.shape) == 3, f"Expected 3D tensor, got {x.shape}"
    assert x.shape[-1] == self.hidden_size, f"Expected hidden_size {self.hidden_size}, got {x.shape[-1]}"
    
    # Continue with forward pass
    ...
```

### Tip 2: Check for NaN/Inf

```python
def check_nan(tensor, name):
    if torch.isnan(tensor).any():
        raise ValueError(f"{name} contains NaN!")
    if torch.isinf(tensor).any():
        raise ValueError(f"{name} contains Inf!")

# Use in training
output = model(input)
check_nan(output, "model_output")

loss = compute_loss(output)
check_nan(loss, "loss")
```

### Tip 3: Gradient Inspection

```python
def inspect_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm = {grad_norm:.4f}")
            
            if grad_norm > 1000:
                print(f"  WARNING: Large gradient!")
            if grad_norm < 1e-7:
                print(f"  WARNING: Vanishing gradient!")

# After backward pass
loss.backward()
inspect_gradients(model)
```

### Tip 4: Overfitting on Small Batch

```python
# Debugging strategy: Overfit on single batch
# If this doesn't work, there's a bug!

def debug_overfit():
    model = LLMModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Single batch
    batch = next(iter(dataloader))
    
    # Train until loss near zero
    for step in range(1000):
        output = model(batch)
        loss = compute_loss(output)
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")
    
    # Loss should be < 0.01
    # If not, there's a bug in model or training loop
```

### Tip 5: Compare with Simple Baseline

```python
# Implement simple version first
class SimpleAttention(nn.Module):
    def forward(self, x):
        # Just average all positions (no learned parameters)
        return x.mean(dim=1, keepdim=True).expand_as(x)

# If this trains successfully, gradually add complexity
# 1. Simple attention ✓
# 2. Multi-head attention
# 3. Add RoPE
# 4. Add GQA
# etc.
```

---

## Further Reading

### PyTorch Documentation
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Autograd Mechanics](https://pytorch.org/docs/stable/notes/autograd.html)
- [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

### Research Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer
- [BERT](https://arxiv.org/abs/1810.04805) - Pre-training methods
- [GPT-2](https://openai.com/research/better-language-models) - Scaling language models
- [GPT-3](https://arxiv.org/abs/2005.14165) - Few-shot learning
- [RoFormer](https://arxiv.org/abs/2104.09864) - RoPE
- [GLU Variants](https://arxiv.org/abs/2002.05202) - SwiGLU
- [Flash Attention](https://arxiv.org/abs/2205.14135) - Efficient attention
- [GQA](https://arxiv.org/abs/2305.13245) - Grouped Query Attention

### Books
- "Deep Learning" by Goodfellow, Bengio, Courville
- "Natural Language Processing with Transformers" by Tunstall et al.
- "Dive into Deep Learning" (free online)

### Video Courses
- Stanford CS224N - NLP with Deep Learning
- Stanford CS231N - Convolutional Neural Networks
- Fast.ai - Practical Deep Learning

---

## Glossary of Terms

### A-D
- **Activation Function:** Non-linear function applied to neuron outputs
- **Adam:** Adaptive learning rate optimization algorithm
- **Attention:** Mechanism to focus on relevant parts of input
- **Autograd:** Automatic differentiation system in PyTorch
- **Backpropagation:** Algorithm for computing gradients
- **Batch:** Group of examples processed together
- **Bias:** Learnable offset parameter in layers
- **Cross-Entropy:** Loss function for classification
- **Distributed Training:** Training across multiple devices

### E-L
- **Embedding:** Dense vector representation of discrete tokens
- **Epoch:** One complete pass through training data
- **Forward Pass:** Computing output from input
- **Gradient:** Partial derivative of loss w.r.t. parameters
- **Hidden State:** Intermediate representation in network
- **Layer Normalization:** Normalization technique for stabilizing training
- **Learning Rate:** Step size for parameter updates
- **Logits:** Raw output scores before softmax

### M-R
- **Mixed Precision:** Using FP16 and FP32 for training
- **Momentum:** Accumulation of gradients over time
- **Multi-Head Attention:** Parallel attention with different projections
- **Optimizer:** Algorithm for updating parameters
- **Perplexity:** Measure of model uncertainty (exp(loss))
- **ReLU:** Rectified Linear Unit activation (max(0, x))
- **Residual Connection:** Skip connection (x + layer(x))
- **RoPE:** Rotary Position Embedding

### S-Z
- **Softmax:** Convert scores to probabilities
- **Tensor:** Multi-dimensional array
- **Tokenization:** Converting text to numerical IDs
- **Transformer:** Neural network architecture based on attention
- **Validation:** Evaluating model on held-out data
- **Weight Decay:** L2 regularization technique
- **Zero Grad:** Clearing accumulated gradients

---

## Quick Reference

### Common Shapes in Our Code

```python
# Input
input_ids: [batch_size, seq_len]

# After embedding
embedded: [batch_size, seq_len, hidden_size]

# In attention (after head split)
q, k, v: [batch_size, num_heads, seq_len, head_dim]

# Attention scores
scores: [batch_size, num_heads, seq_len, seq_len]

# After concatenating heads
attn_output: [batch_size, seq_len, hidden_size]

# Final logits
logits: [batch_size, seq_len, vocab_size]

# Loss
loss: scalar (single number)
```

### Common Operations

```python
# Reshape
x.view(new_shape)
x.reshape(new_shape)

# Transpose
x.transpose(dim1, dim2)
x.permute(dims)

# Matrix multiplication
x @ y
torch.matmul(x, y)

# Element-wise operations
x + y, x * y, x / y

# Aggregations
x.mean(), x.sum(), x.max()

# Indexing
x[0]        # First element
x[:, 0]     # First column
x[0, :5]    # First row, first 5 elements
```

### Common Pitfalls

1. **Forgetting `.to(device)`:**
```python
# Wrong
output = model(input)  # input on CPU, model on GPU

# Right
input = input.to(device)
output = model(input)
```

2. **Not clearing gradients:**
```python
# Wrong
loss.backward()
optimizer.step()
# Gradients accumulate!

# Right
loss.backward()
optimizer.step()
optimizer.zero_grad()
```

3. **Wrong shape for loss:**
```python
# Wrong
loss = F.cross_entropy(logits, labels)  # logits: [32, 10, 5000]

# Right
loss = F.cross_entropy(
    logits.view(-1, 5000),  # [320, 5000]
    labels.view(-1)          # [320]
)
```

4. **Train vs Eval mode:**
```python
# Training
model.train()  # Enables dropout, etc.
output = model(input)

# Evaluation
model.eval()   # Disables dropout
with torch.no_grad():  # Don't compute gradients
    output = model(input)
```

---

This guide covers the essential PyTorch knowledge and theory needed to understand the LLM training framework. Practice the exercises, experiment with the code, and refer back to specific sections as needed!