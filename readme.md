# Modern LLM Training Framework

An Experimental, scalable framework for training large language models from scratch. Includes state-of-the-art architectural improvements and supports training on everything from a single CPU to multi-node GPU clusters.

## 🌟 Features

### Architecture
- ✅ **RoPE (Rotary Position Embeddings)** - Better position encoding
- ✅ **SwiGLU Activation** - Improved over ReLU/GELU  
- ✅ **RMSNorm** - More efficient than LayerNorm
- ✅ **Grouped Query Attention (GQA)** - Reduces memory footprint
- ✅ **Flash Attention** - Memory-efficient attention (when available)
- ✅ **Pre-normalization** - Improved training stability

### Training Capabilities
- ✅ Single CPU/GPU training
- ✅ Multi-GPU training (DDP)
- ✅ Multi-node distributed training
- ✅ Mixed precision training (FP16/BF16)
- ✅ Gradient accumulation
- ✅ Automatic checkpointing 
- ✅ PyTorch 2.0 compile support

### Data Processing
- ✅ Multi-format support (.txt, .md, .json)
- ✅ Multiple tokenizer types (BPE, WordPiece, TikToken)
- ✅ Automatic train/validation split
- ✅ Overlapping chunks for context preservation
- ✅ Multiprocessing for fast preparation

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA Toolkit 11.8+ (for GPU training)
- 8GB+ RAM (16GB+ recommended)

### Setup

```bash
# Clone or download the framework files
mkdir llm_training && cd llm_training

# Install dependencies
pip install -r requirements.txt

# For GPU support (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python utils.py setup
```

## 🚀 Quick Start

### 1. Prepare Your Data

```bash
# Put your text files in a folder
mkdir raw_text_data
# Add your .txt, .md, or .json files

# Process the data
python data_prep_tool.py \
    --input_dir ./raw_text_data \
    --output_dir ./processed_data \
    --vocab_size 32000 \
    --max_length 2048
```

### 2. Train Your Model

**Tiny model (testing):**
```bash
python train.py \
    --data_dir ./processed_data \
    --hidden_size 256 \
    --num_layers 6 \
    --batch_size 8 \
    --max_steps 10000
```

**Small model (single GPU):**
```bash
python train.py \
    --data_dir ./processed_data \
    --hidden_size 768 \
    --num_layers 12 \
    --batch_size 32 \
    --max_steps 100000 \
    --mixed_precision
```

**Multi-GPU training:**
```bash
torchrun --nproc_per_node=4 train.py \
    --data_dir ./processed_data \
    --hidden_size 1024 \
    --num_layers 16 \
    --batch_size 128 \
    --micro_batch_size 32 \
    --mixed_precision
```

### 3. Monitor Training

```bash
# Watch training progress
python utils.py monitor --checkpoint_dir ./checkpoints
 
```

## 📚 Documentation

### File Structure

```
llm_training/
├── data_prep_tool.py      # Data preprocessing tool
├── model.py            # Model architecture
├── train.py            # Training framework
├── utils.py            # Utility scripts
├── requirements.txt    # Dependencies
├── README.md           # This file
│
├── raw_text_data/      # Your source text files
├── processed_data/     # Tokenized training data
│   ├── tokenizer.json
│   ├── dataset/
│   └── metadata.json
└── checkpoints/        # Model checkpoints
    ├── checkpoint_step_1000.pt
    ├── checkpoint_step_2000.pt
    └── ...
```

### Model Configurations

| Size | Hidden | Layers | Heads | Params | GPU Memory | Training Time* |
|------|--------|--------|-------|--------|------------|----------------|
| Tiny | 256 | 6 | 8 | 20M | 2GB | 2 days |
| Small | 768 | 12 | 12 | 125M | 8GB | 5 days |
| Medium | 1024 | 24 | 16 | 350M | 16GB | 2 weeks |
| Large | 1536 | 24 | 16 | 800M | 24GB | 1 month |
| XL | 2048 | 32 | 32 | 1.5B | 40GB | 2 months |

*On single RTX 3090 with mixed precision

### Command Line Arguments

#### data_prep_tool.py

```bash
--input_dir          # Directory with text files (required)
--output_dir         # Output directory (required)
--tokenizer_type     # bpe, wordpiece, or tiktoken (default: bpe)
--vocab_size         # Vocabulary size (default: 32000)
--max_length         # Max sequence length (default: 2048)
--stride             # Overlap between chunks (default: 1024)
--num_workers        # CPU workers for processing (default: 4)
```

#### train.py

**Model Architecture:**
```bash
--vocab_size         # Vocabulary size (default: 32000)
--hidden_size        # Hidden dimension (default: 768)
--num_layers         # Number of layers (default: 12)
--num_heads          # Attention heads (default: 12)
--num_kv_heads       # KV heads for GQA (default: same as num_heads)
```

**Training:**
```bash
--data_dir           # Processed data directory (required)
--save_dir           # Checkpoint save directory (default: ./checkpoints)
--batch_size         # Total batch size (default: 32)
--micro_batch_size   # Batch size per step (for gradient accumulation)
--max_steps          # Training steps (default: 100000)
--learning_rate      # Learning rate (default: 3e-4)
--warmup_steps       # Warmup steps (default: 2000)
--grad_clip          # Gradient clipping (default: 1.0)
```

**System:**
```bash
--mixed_precision    # Enable FP16 training
--compile_model      # Use torch.compile (PyTorch 2.0+)
--checkpoint         # Resume from checkpoint
```

**Logging:**
```bash 
--log_interval       # Steps between logs (default: 100)
--eval_interval      # Steps between validation (default: 500)
--