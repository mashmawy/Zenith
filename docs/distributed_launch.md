# LLM Training Framework Documentation

## Overview

This is a modern, scalable LLM training framework with state-of-the-art features:

### Key Features

**Architecture Improvements:**
- ✅ **RoPE (Rotary Position Embeddings)** - Better position encoding than absolute/learned embeddings
- ✅ **SwiGLU Activation** - Superior to ReLU/GELU in FFN layers
- ✅ **RMSNorm** - More efficient than LayerNorm
- ✅ **Grouped Query Attention (GQA)** - Reduces KV cache while maintaining quality
- ✅ **Flash Attention** - Memory-efficient attention computation
- ✅ **Pre-normalization** - More stable training

**Training Features:**
- ✅ Distributed training (DDP) for multi-GPU and multi-node
- ✅ Mixed precision training (FP16/BF16)
- ✅ Gradient accumulation
- ✅ Gradient clipping
- ✅ Cosine learning rate schedule with warmup
- ✅ AdamW optimizer with proper weight decay
- ✅ Checkpointing and resuming 
- ✅ PyTorch 2.0 torch.compile support

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Prepare Data

First, prepare your training data from raw text files:

```bash
python data_prep_tool.py \
    --input_dir ./raw_text_data \
    --output_dir ./processed_data \
    --tokenizer_type bpe \
    --vocab_size 32000 \
    --max_length 2048 \
    --num_workers 8
```

**Options:**
- `--tokenizer_type`: Choose from `bpe`, `wordpiece`, or `tiktoken`
- `--vocab_size`: Size of vocabulary (only for bpe/wordpiece)
- `--max_length`: Maximum sequence length
- `--stride`: Overlap between chunks (for context continuity)

### 2. Train Model

#### Single GPU Training

```bash
python train.py \
    --data_dir ./processed_data \
    --save_dir ./checkpoints \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 32 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --mixed_precision
```

#### Multi-GPU Training (Single Node)

```bash
torchrun --nproc_per_node=4 train.py \
    --data_dir ./processed_data \
    --save_dir ./checkpoints \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 128 \
    --micro_batch_size 32 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --mixed_precision
```

#### Multi-Node Training

**On each node, run:**

```bash
# Node 0 (master)
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py \
    --data_dir ./processed_data \
    --save_dir ./checkpoints \
    --hidden_size 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --batch_size 512 \
    --micro_batch_size 16 \
    --max_steps 500000 \
    --learning_rate 3e-4 \
    --mixed_precision

# Node 1
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=29500 \
    train.py [same args...]

# Repeat for nodes 2 and 3 with node_rank=2 and node_rank=3
```

#### CPU Training (for testing/small models)

```bash
python train.py \
    --data_dir ./processed_data \
    --save_dir ./checkpoints \
    --hidden_size 256 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 8 \
    --max_steps 10000 \
    --learning_rate 3e-4
```

### 3. Resume Training

```bash
python train.py \
    --data_dir ./processed_data \
    --checkpoint ./checkpoints/checkpoint_step_50000.pt \
    [other args...]
```

## Model Configurations

### Tiny Model (Testing)
```bash
--hidden_size 256 --num_layers 6 --num_heads 8
# ~20M parameters
```

### Small Model
```bash
--hidden_size 768 --num_layers 12 --num_heads 12
# ~125M parameters (GPT-2 Small size)
```

### Medium Model
```bash
--hidden_size 1024 --num_layers 24 --num_heads 16 --num_kv_heads 8
# ~350M parameters with GQA
```

### Large Model
```bash
--hidden_size 1536 --num_layers 24 --num_heads 16 --num_kv_heads 8
# ~800M parameters with GQA
```

### XL Model
```bash
--hidden_size 2048 --num_layers 32 --num_heads 32 --num_kv_heads 8
# ~1.5B parameters with GQA
```

## Advanced Features

### Grouped Query Attention (GQA)

Reduce KV cache size while maintaining quality:

```bash
--num_heads 16 --num_kv_heads 4  # 4x reduction in KV cache
```

### Gradient Accumulation

Train with larger effective batch sizes:

```bash
--batch_size 128 --micro_batch_size 32  # Accumulate over 4 steps
```

### Mixed Precision Training

Enable for 2x faster training and 2x less memory:

```bash
--mixed_precision
```

### Model Compilation (PyTorch 2.0+)

Enable torch.compile for additional speedup:

```bash
--compile_model
```
 
## Project Structure

```
.
├── data_prep_tool.py      # Data preparation tool
├── model.py            # Model architecture
├── train.py            # Training framework
├── requirements.txt    # Dependencies
├── raw_text_data/      # Your raw text files
├── processed_data/     # Tokenized dataset
│   ├── tokenizer.json
│   ├── dataset/
│   └── metadata.json
└── checkpoints/        # Model checkpoints
```

## Performance Tips

1. **Batch Size**: Largest that fits in memory (use gradient accumulation if needed)
2. **Mixed Precision**: Always use on modern GPUs (A100, H100, RTX 30xx+)
3. **Flash Attention**: Automatically enabled when available
4. **Gradient Checkpointing**: Add if running out of memory (trade compute for memory)
5. **Data Loading**: Increase `num_workers` in DataLoader for faster I/O
6. **Model Compilation**: Use `--compile_model` with PyTorch 2.0+ for 10-20% speedup

## Monitoring

Check training progress:
- Console logs show loss, learning rate, and throughput 
- Validation loss every `--eval_interval` steps

## Troubleshooting

### Out of Memory
- Reduce `--micro_batch_size`
- Reduce `--max_length` in data preparation
- Enable mixed precision with `--mixed_precision`
- Use smaller model size

### Slow Training
- Enable `--mixed_precision`
- Use `--compile_model` (PyTorch 2.0+)
- Increase `--batch_size` (with gradient accumulation)
- Check if Flash Attention is enabled (automatic)

### Distributed Training Issues
- Ensure all nodes can communicate
- Check firewall settings for master_port
- Verify same code version on all nodes
- Check NCCL environment variables

## Citation

This framework implements techniques from:
- RoPE: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- SwiGLU: "GLU Variants Improve Transformer"
- RMSNorm: "Root Mean Square Layer Normalization"
- GQA: "GQA: Training Generalized Multi-Query Transformer Models"
- Flash Attention: "FlashAttention: Fast and Memory-Efficient Exact Attention"
