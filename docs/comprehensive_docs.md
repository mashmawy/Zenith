# Complete LLM Training Framework Guide
## For Complete Beginners (No Python Experience Required)

---

## Table of Contents
1. [What is This?](#what-is-this)
2. [Prerequisites](#prerequisites)
3. [Installation Guide](#installation-guide)
4. [Understanding the Components](#understanding-the-components)
5. [Step-by-Step Tutorial](#step-by-step-tutorial)
6. [Code Explanation](#code-explanation)
7. [Troubleshooting](#troubleshooting)
8. [Glossary](#glossary)

---

## What is This?

This is a framework (a collection of code) that allows you to train your own AI language model (like ChatGPT, but smaller). Think of it as a complete recipe and kitchen for baking an AI model from scratch.

**What can you do with this?**
- Train a small AI model on your own computer
- Train a large AI model on multiple powerful computers
- Process text data from any source (books, articles, websites)
- Create custom AI models for specific tasks

---

## Prerequisites

### What You Need

**Hardware (at minimum):**
- A computer with at least 8GB RAM
- For GPU training: An NVIDIA graphics card (recommended: RTX 3060 or better)
- For serious training: Multiple GPUs or cloud computing (AWS, Google Cloud)

**Software:**
- Python 3.8 or newer (programming language)
- CUDA Toolkit (for GPU support - only if you have NVIDIA GPU)
- Text editor or IDE (VS Code recommended)

**Knowledge:**
- Basic computer skills (file management, command line)
- This guide will teach you everything else!

---

## Installation Guide

### Step 1: Install Python

**Windows:**
1. Go to https://www.python.org/downloads/
2. Download Python 3.10 or newer
3. Run the installer
4. ⚠️ **IMPORTANT**: Check the box "Add Python to PATH"
5. Click "Install Now"

**Mac/Linux:**
```bash
# Mac (using Homebrew)
brew install python3

# Linux (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install python3 python3-pip
```

### Step 2: Verify Installation

Open your terminal/command prompt and type:
```bash
python --version
```
You should see something like: `Python 3.10.x`

### Step 3: Install CUDA (For GPU Training)

**Only if you have NVIDIA GPU:**
1. Go to https://developer.nvidia.com/cuda-downloads
2. Download CUDA Toolkit 11.8 or 12.1
3. Install following the instructions
4. Restart your computer

### Step 4: Create Project Folder

```bash
# Create a folder for your project
mkdir my_llm_project
cd my_llm_project

# Create subfolders
mkdir raw_text_data
mkdir processed_data
mkdir checkpoints
```

### Step 5: Download the Code Files

Place these files in your `my_llm_project` folder:
- `data_prep_tool.py`
- `model.py`
- `train.py`
- `requirements.txt`

### Step 6: Install Required Packages

Open terminal in your project folder and run:
```bash
pip install -r requirements.txt
```

This installs all the libraries (pre-written code) that our framework needs.

**What gets installed:**
- `torch`: PyTorch - the main deep learning library
- `tokenizers`: Converts text to numbers
- `datasets`: Manages training data 
⏱️ This may take 5-15 minutes depending on your internet speed.

---

## Understanding the Components

### The Three Main Files

#### 1. `data_prep_tool.py` - The Data Chef

**What it does:**
Converts your raw text files into a format the AI can learn from.

**Think of it like:**
A chef preparing ingredients - washing, cutting, organizing them before cooking.

**Input:** 
- Folder with text files (.txt, .md, .json)
- Example: Books, articles, code, conversations

**Output:**
- `tokenizer.json` - Dictionary that converts words to numbers
- `dataset/` folder - Processed, ready-to-train data
- `metadata.json` - Information about your data

#### 2. `model.py` - The Brain Blueprint

**What it does:**
Defines the architecture (structure) of your AI model.

**Think of it like:**
The blueprint for building a brain - how neurons connect, how information flows.

**Key Components:**
- **RMSNorm**: Keeps numbers stable during training
- **RotaryEmbedding**: Helps the model understand word positions
- **GroupedQueryAttention**: Makes the model pay attention to important words
- **SwiGLU**: Activation function (like neurons firing)
- **TransformerBlock**: One layer of the brain
- **LLMModel**: The complete brain with all layers

#### 3. `train.py` - The Teacher

**What it does:**
Actually trains your AI model by showing it data repeatedly.

**Think of it like:**
A teacher showing flashcards to a student, correcting mistakes, and testing progress.

**Key Functions:**
- Loads your prepared data
- Creates the model brain
- Trains it by showing examples
- Saves progress (checkpoints)
- Evaluates performance
- Handles multiple computers if available

---

## Step-by-Step Tutorial

### Mission: Train Your First Language Model

#### Phase 1: Prepare Your Data

**Step 1: Collect Text Data**

Put text files in the `raw_text_data` folder. Examples:
- Download books from Project Gutenberg
- Copy articles or documents
- Use your own writing
- Code repositories

**How much data?**
- Minimum: 10MB (for tiny model)
- Small model: 100MB - 1GB
- Good model: 10GB+
- Professional model: 100GB+

**Step 2: Run Data Preparation**

Open terminal in your project folder:

```bash
python data_prep_tool.py \
    --input_dir ./raw_text_data \
    --output_dir ./processed_data \
    --tokenizer_type bpe \
    --vocab_size 32000 \
    --max_length 2048 \
    --num_workers 4
```

**Let's understand each option:**

- `--input_dir ./raw_text_data`
  - Where your text files are
  - `./` means "current folder"

- `--output_dir ./processed_data`
  - Where to save processed data

- `--tokenizer_type bpe`
  - Type of text splitting method
  - Options: `bpe` (recommended), `wordpiece`, `tiktoken`
  - BPE = Byte Pair Encoding (efficient for most languages)

- `--vocab_size 32000`
  - Size of the vocabulary (dictionary)
  - Larger = more words but slower training
  - 32000 is a good default

- `--max_length 2048`
  - Maximum number of tokens (word pieces) per example
  - Longer = more context but more memory needed
  - 2048 is standard

- `--num_workers 4`
  - Number of CPU cores to use
  - More = faster processing
  - Don't exceed your CPU core count

**What happens:**
```
Found 150 files
Reading files: 100%|████████████| 150/150
Training bpe tokenizer...
Tokenizing and chunking files: 100%|████████████| 150/150
Created 45,632 training examples
Train examples: 43,350
Validation examples: 2,282
```

⏱️ **Time estimate:** 5-30 minutes depending on data size

#### Phase 2: Train Your Model

**Step 3: Choose Model Size**

First, let's understand model sizes and requirements:

| Model Size | Parameters | RAM Needed | GPU Memory | Training Time* |
|------------|-----------|------------|------------|----------------|
| Tiny       | 20M       | 4GB        | 2GB        | 1-2 days       |
| Small      | 125M      | 8GB        | 8GB        | 3-5 days       |
| Medium     | 350M      | 16GB       | 16GB       | 1-2 weeks      |
| Large      | 800M      | 32GB       | 24GB       | 2-4 weeks      |
| XL         | 1.5B      | 64GB       | 40GB       | 1-2 months     |

*On single GPU (RTX 3090 equivalent)

**Step 4: Training Command**

**For Tiny Model (Testing - CPU OK):**
```bash
python train.py \
    --data_dir ./processed_data \
    --save_dir ./checkpoints \
    --hidden_size 256 \
    --num_layers 6 \
    --num_heads 8 \
    --batch_size 8 \
    --max_steps 10000 \
    --learning_rate 3e-4 \
    --warmup_steps 500
```

**For Small Model (Single GPU):**
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
    --warmup_steps 2000 \
    --mixed_precision
```

**Let's understand each parameter:**

**Model Architecture:**
- `--hidden_size 768`
  - Width of the neural network
  - Larger = more capacity but slower
  - Common values: 256, 512, 768, 1024, 2048

- `--num_layers 12`
  - Depth of the neural network (number of transformer blocks)
  - More layers = better understanding but slower
  - Common values: 6, 12, 24, 32

- `--num_heads 12`
  - Number of attention heads (parallel processors)
  - Should divide evenly into hidden_size
  - Common values: 8, 12, 16, 32

- `--num_kv_heads 4` (optional)
  - For Grouped Query Attention (advanced)
  - Saves memory during inference
  - Omit this for standard training

**Training Parameters:**
- `--batch_size 32`
  - Number of examples to process at once
  - Larger = faster but needs more memory
  - Start with 8, increase until memory is full

- `--micro_batch_size 8` (optional)
  - Split batch into smaller chunks
  - Use when running out of memory
  - batch_size / micro_batch_size = accumulation steps

- `--max_steps 100000`
  - Total training iterations
  - More steps = better quality (to a point)
  - 100k steps ≈ seeing all data 2-3 times

- `--learning_rate 3e-4`
  - How fast the model learns
  - Too high = unstable, too low = slow
  - 3e-4 = 0.0003 (standard value)

- `--warmup_steps 2000`
  - Gradually increase learning rate at start
  - Prevents early instability
  - Usually 2-5% of max_steps

- `--grad_clip 1.0`
  - Prevents gradients from exploding
  - 1.0 is standard, no need to change

**System Options:**
- `--mixed_precision`
  - Use 16-bit floating point (faster, less memory)
  - Always use with modern GPUs
  - 2x speedup typically

- `--compile_model`
  - PyTorch 2.0+ optimization
  - Additional 10-20% speedup
  - Only works with PyTorch 2.0+
 

**What you'll see:**
```
============================================================
Training Configuration
============================================================
Device: cuda:0
Distributed: False (World size: 1)
Model parameters: 124,439,296
Batch size: 32 (micro: 32)
Gradient accumulation steps: 1
Mixed precision: True
============================================================

Step 100/100000 | Loss: 8.2341 | LR: 1.50e-05 | Tokens/sec: 12453
Step 200/100000 | Loss: 7.8932 | LR: 3.00e-05 | Tokens/sec: 12621
Validation Loss: 7.6543
...
```

**Understanding the output:**
- **Loss**: How wrong the model is (lower = better)
  - Start: 8-10
  - Good: 2-4
  - Great: <2
- **LR**: Current learning rate (changes during training)
- **Tokens/sec**: Training speed

#### Phase 3: Monitor Training

**Step 5: Watch Progress**

**In Terminal:**
- Loss should decrease over time
- If loss increases consistently, something is wrong
 
**Checkpoints:**
Every `--save_interval` steps (default 1000), a checkpoint is saved:
```
checkpoints/
  checkpoint_step_1000.pt
  checkpoint_step_2000.pt
  ...
```

**Step 6: Resume if Interrupted**

If training stops (computer crash, power outage), resume:
```bash
python train.py \
    --checkpoint ./checkpoints/checkpoint_step_5000.pt \
    [... all other parameters same as before ...]
```

---

## Code Explanation

### Detailed Walkthrough

#### Understanding `data_prep_tool.py`

```python
# Import necessary libraries
import os  # For file operations
import json  # For saving/loading JSON files
from pathlib import Path  # Modern way to handle file paths
from typing import List  # For type hints (documentation)
```

**Key Class: DataPreparator**

```python
class DataPreparator:
    """Handles data collection, tokenization, and dataset creation"""
```

A "class" is like a blueprint. DataPreparator is a blueprint for an object that prepares data.

**Important Methods:**

1. **collect_files()**
```python
def collect_files(self) -> List[Path]:
    """Recursively collect all text files from input directory"""
```
- Searches for all .txt, .md, .json files
- "Recursively" means it looks in subfolders too
- Returns a list of file paths

2. **train_tokenizer()**
```python
def train_tokenizer(self, files: List[Path]):
    """Train or load tokenizer"""
```
- Creates a "tokenizer" - converts text to numbers
- Tokenizer learns common words/patterns from your data
- Saves it for reuse

**Why tokenization?**
- Computers can't understand words directly
- We convert: "Hello world" → [15496, 995]
- Each number represents a word or part of a word

3. **create_chunks()**
```python
def create_chunks(self, token_ids: List[int]) -> List[List[int]]:
    """Split token sequence into overlapping chunks"""
```
- Takes long text and splits into smaller pieces
- Overlapping ensures no context is lost at boundaries
- Example: "The quick brown fox jumps over the lazy dog"
  - Chunk 1: "The quick brown fox jumps"
  - Chunk 2: "fox jumps over the lazy dog"
  - Note "fox jumps" appears in both (overlap)

#### Understanding `model.py`

**The Transformer Architecture:**

```python
class LLMModel(nn.Module):
    """Complete Language Model"""
```

This is your AI brain. Let's break it down:

**1. Embeddings**
```python
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```
- Converts token numbers to vectors (lists of numbers)
- Each word becomes a point in high-dimensional space
- Similar words become nearby points

**2. Transformer Layers**
```python
self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
```
- Stack of transformer blocks (the actual "thinking" layers)
- Each layer processes information and passes to next
- Like a chain of specialists, each adding understanding

**Inside a TransformerBlock:**

```python
class TransformerBlock(nn.Module):
    def __init__(self, config):
        self.attention = GroupedQueryAttention(config)  # Focus mechanism
        self.feed_forward = SwiGLU(config)  # Processing unit
        self.attention_norm = RMSNorm(config.hidden_size)  # Stabilizer
        self.ffn_norm = RMSNorm(config.hidden_size)  # Stabilizer
```

- **Attention**: Looks at all words to understand context
  - "Bank" near "river" vs "bank" near "money"
- **Feed Forward**: Processes each word individually
- **Normalization**: Keeps numbers in reasonable range

**Key Innovation: Rotary Position Embedding (RoPE)**

```python
class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding for better position encoding"""
```

- Helps model understand word order
- "Dog bites man" vs "Man bites dog" - different meanings!
- RoPE is better than older methods (learned positions)

**Key Innovation: Grouped Query Attention (GQA)**

```python
self.num_kv_groups = self.num_heads // self.num_kv_heads
```

- Standard attention: Every head has its own keys and values
- GQA: Multiple heads share keys and values
- Result: Much less memory, almost same quality
- Example: 16 heads with 4 kv_heads = 4x memory savings

#### Understanding `train.py`

**The Training Loop:**

```python
def train(self):
    """Main training loop"""
    for batch in self.train_loader:
        loss = self.train_step(batch)  # Process one batch
        loss.backward()  # Calculate gradients
        self.optimizer.step()  # Update model weights
```

**What's happening:**

1. **Forward Pass**: Model makes predictions
```python
logits, loss = self.model(input_ids, labels=input_ids)
```
- Input: Token IDs [15496, 995, 389, ...]
- Output: Predictions for next token
- Loss: How wrong the predictions are

2. **Backward Pass**: Calculate how to improve
```python
loss.backward()
```
- Uses calculus to find gradient (direction to improve)
- Gradient tells us: "Move weights this way to reduce loss"

3. **Optimizer Step**: Actually improve the model
```python
self.optimizer.step()
```
- Updates all model weights slightly
- Moves in direction that reduces loss

**Learning Rate Schedule:**

```python
def get_lr(step: int, config: TrainingConfig) -> float:
    """Cosine learning rate schedule with warmup"""
```

- **Warmup**: Start with small learning rate, increase gradually
  - Prevents early instability
- **Cosine Decay**: Slowly decrease learning rate
  - Fine-tunes the model as it improves
  
**Think of it like:**
- Learning to drive: Start slow (warmup), then normal speed, then careful adjustments (decay)

**Mixed Precision Training:**

```python
with autocast(enabled=(self.scaler is not None)):
    logits, loss = self.model(...)
```

- Normally: Uses 32-bit floating point numbers
- Mixed precision: Uses 16-bit for most operations
- Result: 2x faster, uses half the memory
- Quality: Almost identical results

**Distributed Training:**

```python
if self.config.distributed:
    model = DDP(model, device_ids=[self.local_rank])
```

- DDP = DistributedDataParallel
- Splits work across multiple GPUs
- Each GPU processes different batch
- Gradients are averaged across all GPUs

**Flow:**
1. GPU 0 processes batch A, calculates gradient A
2. GPU 1 processes batch B, calculates gradient B
3. GPUs average: (A + B) / 2
4. All GPUs update weights with averaged gradient
5. Repeat

---

## Advanced Usage

### Multi-GPU Training (Single Machine)

If you have multiple GPUs:

```bash
# Check how many GPUs you have
nvidia-smi

# Use torchrun to launch
torchrun --nproc_per_node=4 train.py \
    --data_dir ./processed_data \
    --batch_size 128 \
    --micro_batch_size 32 \
    --mixed_precision
```

**What's happening:**
- `--nproc_per_node=4`: Use 4 GPUs
- Framework automatically splits work
- 4x faster training (approximately)

### Multi-Node Training (Multiple Machines)

For serious training with multiple computers:

**Requirements:**
- All machines connected on same network
- Same code and data on all machines
- Port 29500 open (or choose different)

**On Master Node (Machine 1):**
```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py [args...]
```

**On Worker Nodes (Machines 2-4):**
```bash
# Machine 2
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr="192.168.1.100" \
    --master_port=29500 \
    train.py [args...]

# Machine 3: node_rank=2
# Machine 4: node_rank=3
```

**Parameters explained:**
- `--nproc_per_node=8`: 8 GPUs per machine
- `--nnodes=4`: 4 machines total
- `--node_rank=X`: Machine ID (0 to 3)
- `--master_addr`: IP of master machine
- `--master_port`: Communication port

---

## Troubleshooting

### Common Errors and Solutions

#### 1. "CUDA out of memory"

**Problem:** GPU doesn't have enough memory

**Solutions:**
```bash
# Reduce batch size
--batch_size 16  # Instead of 32

# Use gradient accumulation
--batch_size 32 --micro_batch_size 8  # Process 8 at a time, accumulate to 32

# Reduce model size
--hidden_size 512 --num_layers 8  # Smaller model

# Reduce sequence length
# Go back to data preparation:
python data_prep_tool.py --max_length 1024  # Instead of 2048
```

#### 2. "Loss is NaN" or "Loss is increasing"

**Problem:** Training is unstable

**Solutions:**
```bash
# Reduce learning rate
--learning_rate 1e-4  # Instead of 3e-4

# Increase warmup
--warmup_steps 5000  # Instead of 2000

# Check your data
# Make sure processed data isn't corrupted

# Enable mixed precision
--mixed_precision  # Sometimes helps stability
```

#### 3. "ModuleNotFoundError: No module named 'torch'"

**Problem:** PyTorch not installed properly

**Solution:**
```bash
# Reinstall PyTorch
pip uninstall torch
pip install torch torchvision torchaudio

# For GPU support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. "RuntimeError: Address already in use"

**Problem:** Port 29500 (for distributed training) is busy

**Solution:**
```bash
# Use different port
--master_port=29501

# Or kill the process using port 29500
# Windows:
netstat -ano | findstr :29500
taskkill /PID <process_id> /F

# Linux:
lsof -ti:29500 | xargs kill -9
```

#### 5. Training is Very Slow

**Checklist:**
- ✓ Using GPU? Check with `nvidia-smi`
- ✓ Mixed precision enabled? Add `--mixed_precision`
- ✓ Batch size large enough? Increase until memory is full
- ✓ Data on SSD? Move data to faster drive
- ✓ Enough CPU workers? Check `num_workers` in code

#### 6. "Connection refused" in Multi-Node Training

**Problem:** Machines can't communicate

**Solutions:**
```bash
# Check if master is reachable
ping 192.168.1.100

# Check if port is open
telnet 192.168.1.100 29500

# Disable firewall temporarily
# Windows: Turn off Windows Firewall
# Linux: sudo ufw disable

# Use different network interface
# Find your IP: ipconfig (Windows) or ifconfig (Linux)
```

---

## Performance Optimization

### Getting Maximum Speed

1. **Hardware Utilization**
```bash
# Monitor GPU usage
nvidia-smi -l 1  # Update every second

# Target: GPU utilization 95-100%
# If lower, increase batch size
```

2. **Optimal Batch Size**
```python
# Start small and increase until OOM (out of memory)
# Example progression:
--batch_size 8   # Try this
--batch_size 16  # If it works, try this
--batch_size 32  # Keep increasing
--batch_size 64  # Until you get CUDA OOM error
--batch_size 48  # Then go back one step
```

3. **Enable All Optimizations**
```bash
python train.py \
    --mixed_precision \      # 2x speedup
    --compile_model \        # +10-20% speedup (PyTorch 2.0+)
    --data_dir ./processed_data
```

4. **Gradient Accumulation**
```bash
# If batch_size=64 causes OOM, use:
--batch_size 64 --micro_batch_size 16

# This processes 16 at a time, accumulates to 64
# Same result, 4x less memory needed
```

5. **Data Loading**
```python
# In train.py, increase num_workers:
DataLoader(..., num_workers=8)  # Adjust based on CPU cores

# More workers = faster data loading
# Don't exceed: number of CPU cores - 2
```

---

## Glossary

### Key Terms Explained

**AI/ML Terms:**

- **Token**: A piece of text (word, part of word, or character)
  - Example: "Hello world!" → ["Hello", " world", "!"] → [15496, 995, 0]

- **Embedding**: Converting tokens to vectors (lists of numbers)
  - "cat" → [0.2, -0.5, 0.8, 0.1, ...]
  - Similar words have similar vectors

- **Attention**: Mechanism for focusing on relevant parts of input
  - Reading "bank", looks back at "river" or "money" for context

- **Transformer**: Type of neural network architecture
  - Good at understanding sequences (text, audio, etc.)
  - Uses attention mechanisms

- **Loss**: Measure of how wrong the model is
  - Lower = better
  - Training aims to minimize loss

- **Gradient**: Direction to adjust weights to reduce loss
  - Calculated using calculus (backpropagation)

- **Learning Rate**: How much to adjust weights each step
  - Too high: Model is unstable
  - Too low: Learning is slow
  - 3e-4 = 0.0003 is standard

- **Epoch**: One complete pass through the training data
  - 10 epochs = seeing all data 10 times

- **Batch**: Group of examples processed together
  - Batch size 32 = process 32 examples at once

**Technical Terms:**

- **GPU**: Graphics Processing Unit
  - Specialized chip for parallel computations
  - 10-100x faster than CPU for deep learning

- **CUDA**: NVIDIA's parallel computing platform
  - Allows PyTorch to use NVIDIA GPUs
  - AMD GPUs use ROCm instead

- **Mixed Precision**: Using 16-bit and 32-bit floats
  - FP16 for most operations (faster)
  - FP32 for critical operations (accuracy)

- **Distributed Training**: Using multiple GPUs/machines
  - DDP: DistributedDataParallel
  - Splits work across devices

- **Checkpoint**: Saved model state
  - Allows resuming training
  - Also used for deployment

**Architecture Terms:**

- **RoPE**: Rotary Position Embedding
  - Better way to encode word positions
  - Helps model understand order

- **SwiGLU**: Swish-Gated Linear Unit
  - Activation function (like neuron firing)
  - Better than older ReLU

- **RMSNorm**: Root Mean Square Normalization
  - Keeps activations in reasonable range
  - More efficient than LayerNorm

- **GQA**: Grouped Query Attention
  - Memory-efficient attention
  - Multiple heads share keys/values

- **Flash Attention**: Optimized attention computation
  - Much faster and memory-efficient
  - Automatically used if available

**Training Terms:**

- **Forward Pass**: Model makes predictions
  - Input → Model → Output

- **Backward Pass**: Calculate gradients
  - Output → Calculate error → Propagate back

- **Optimizer**: Algorithm that updates weights
  - AdamW is most common
  - Uses gradients to improve model

- **Warmup**: Gradually increasing learning rate
  - Prevents early instability
  - Usually first 2-5% of training

- **Learning Rate Schedule**: Changing learning rate during training
  - Often starts low, increases, then decreases
  - Cosine schedule is popular

---

## FAQ

**Q: How much data do I need?**
A: Minimum 10MB for testing, 1GB+ for useful models, 10GB+ for good quality. More is always better.

**Q: How long does training take?**
A: Depends on model size and hardware. Small model on one GPU: 3-5 days. Large model on 8 GPUs: 1-2 weeks.

**Q: Can I train on CPU?**
A: Yes, but it's 10-100x slower. Only practical for tiny models or testing.

**Q: What if I don't have a powerful GPU?**
A: Use cloud services: Google Colab (free tier), AWS, Google Cloud, or Lambda Labs.

**Q: How do I know if my model is good?**
A: Watch validation loss. If it stops decreasing, model has learned all it can from your data.

**Q: Can I stop and resume training?**
A: Yes! Use `--checkpoint ./checkpoints/checkpoint_step_5000.pt` to resume.

**Q: What's the difference between parameters?**
A: Parameters are the learned weights. More parameters = more capacity to learn, but slower and needs more data.

**Q: Should I use mixed precision?**
A: Yes, always use it with modern GPUs (anything newer than GTX 1000 series).

**Q: What's gradient accumulation?**
A: Processing smaller batches but accumulating gradients as if it were a larger batch. Saves memory.

**Q: How do I choose hyperparameters?**
A: Start with defaults in this guide. If loss doesn't decrease, try lower learning rate or more warmup.

---

## Next Steps

After successfully training your model:

1. **Evaluation**: Test model quality on unseen data
2. **Fine-tuning**: Specialize model for specific tasks
3. **Deployment**: Serve model for inference
4. **Optimization**: Quantization, distillation for faster inference

**Resources to Learn More:**
- PyTorch tutorials: pytorch.org/tutorials
- Hugging Face course: huggingface.co/course
- Papers: "Attention is All You Need", "GPT-3", "LLaMA"

---

## Support

If you need help:
1. Check error message in troubleshooting section
2. Search error on Google/Stack Overflow
3. Ask in PyTorch forums or Discord
4. Check GitHub issues if using open-source