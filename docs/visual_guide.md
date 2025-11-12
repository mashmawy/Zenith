# Visual Guide & Practical Examples
## Understanding LLM Training Through Diagrams and Real Scenarios

---

## Part 1: Visual Architecture Overview

### The Complete Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     1. DATA PREPARATION                          │
│                                                                   │
│  Raw Text Files          Tokenization         Training Dataset   │
│  ┌──────────┐           ┌──────────┐         ┌──────────┐      │
│  │ book.txt │           │          │         │ [1,45,2] │      │
│  │article.md│  ───────► │ BPE/WP   │ ─────► │ [8,12,9] │      │
│  │ code.py  │           │ Tokenizer│         │ [3,67,4] │      │
│  └──────────┘           └──────────┘         └──────────┘      │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     2. MODEL ARCHITECTURE                         │
│                                                                   │
│  Input: [1, 45, 2, 89, 12]                                       │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   Embedding     │  Convert tokens to vectors                 │
│  └─────────────────┘                                            │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ Transformer 1   │  ┌──────────────┐                         │
│  │ ┌─────────────┐ │  │ Attention    │                         │
│  │ │  Attention  │ │  │ (focus on    │                         │
│  │ └─────────────┘ │  │  context)    │                         │
│  │ ┌─────────────┐ │  └──────────────┘                         │
│  │ │   SwiGLU    │ │  ┌──────────────┐                         │
│  │ └─────────────┘ │  │ Feed Forward │                         │
│  └─────────────────┘  │ (process)    │                         │
│         │              └──────────────┘                         │
│         ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ Transformer 2   │  (Repeated N times)                        │
│  └─────────────────┘                                            │
│         │                                                         │
│        ...                                                        │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────┐                                            │
│  │ Transformer N   │                                            │
│  └─────────────────┘                                            │
│         │                                                         │
│         ▼                                                         │
│  ┌─────────────────┐                                            │
│  │   Output Head   │  Predict next token                        │
│  └─────────────────┘                                            │
│         │                                                         │
│         ▼                                                         │
│  Predictions: [0.01, 0.85, 0.02, ...]  (probability per token)  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     3. TRAINING LOOP                              │
│                                                                   │
│  ┌──────────┐     ┌──────────┐     ┌──────────┐               │
│  │ Forward  │────►│ Compute  │────►│ Backward │               │
│  │   Pass   │     │   Loss   │     │   Pass   │               │
│  └──────────┘     └──────────┘     └──────────┘               │
│       │                                    │                     │
│       │                                    ▼                     │
│       │                            ┌──────────┐                 │
│       │                            │ Update   │                 │
│       └────────────────────────────│ Weights  │                 │
│                                    └──────────┘                 │
│                                         │                        │
│                                         │                        │
│                                    Repeat 100k+ times            │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### How Attention Works (Simplified)

```
Input Sentence: "The cat sat on the mat"

Step 1: Convert to tokens
┌─────┬─────┬─────┬─────┬─────┬─────┐
│ The │ cat │ sat │ on  │ the │ mat │
└─────┴─────┴─────┴─────┴─────┴─────┘
   │     │     │     │     │     │
   ▼     ▼     ▼     ▼     ▼     ▼
  [5]  [847] [329] [319] [5]  [452]

Step 2: Attention for word "sat"
When processing "sat", attention looks at ALL words:

        ┌─ 0.05 ← "The"    (low attention)
        ├─ 0.60 ← "cat"    (high attention - who sat?)
"sat" ──┼─ 0.20 ← "sat"    (medium - itself)
        ├─ 0.10 ← "on"     (medium attention)
        ├─ 0.03 ← "the"    (low attention)
        └─ 0.02 ← "mat"    (low attention - not relevant yet)

Result: Model understands "cat" is the one sitting!

Step 3: For word "mat"
        ┌─ 0.02 ← "The"
        ├─ 0.15 ← "cat"
"mat" ──┼─ 0.40 ← "sat"    (high - action related to mat)
        ├─ 0.35 ← "on"     (high - preposition indicating location)
        ├─ 0.05 ← "the"
        └─ 0.03 ← "mat"

Result: Model understands relationship between sitting and mat!
```

### Training Process Over Time

```
Training Progress Visualization:

Loss (Lower is Better)
│
10│     ●
  │      ●
  │       ●●
  │         ●●
  │           ●●
 5│             ●●●
  │                ●●●●
  │                    ●●●●●
  │                         ●●●●●●
 2│                               ●●●●●●●●
  │                                      ●●●●●●●
  │────────────────────────────────────────────────►
  0        25k       50k      75k       100k    Steps

Phase 1 (0-10k):    Fast improvement, model learns basics
Phase 2 (10k-50k):  Steady improvement, learning patterns
Phase 3 (50k-100k): Slow improvement, fine-tuning details

Learning Rate Schedule:
│
  │          ╱────────────╲
LR│        ╱                ╲
  │      ╱                    ╲___
  │    ╱                          ────
  │  ╱                                ────
  │╱                                      ────
  │────────────────────────────────────────────►
     Warmup    Training        Decay      Steps
```

---

## Part 2: Practical Examples

### Example 1: Training a Tiny Model (For Testing)

**Scenario:** You want to test if everything works before committing to long training.

**Your Computer:**
- 8GB RAM
- No GPU (CPU only)
- 50MB of text data

**Step-by-Step:**

```bash
# 1. Prepare data (takes ~2 minutes)
python data_prep_tool.py \
    --input_dir ./my_texts \
    --output_dir ./processed \
    --vocab_size 8000 \
    --max_length 512 \
    --num_workers 2

# 2. Train tiny model (takes ~1 hour)
python train.py \
    --data_dir ./processed \
    --hidden_size 128 \
    --num_layers 4 \
    --num_heads 4 \
    --batch_size 4 \
    --max_steps 1000 \
    --learning_rate 3e-4

# Expected output:
# Step 100/1000 | Loss: 6.2341 | LR: 3.00e-05
# Step 200/1000 | Loss: 5.8932 | LR: 6.00e-05
# ...
# Step 1000/1000 | Loss: 3.2456 | LR: 3.00e-04
```

**Result:** 
- Model parameters: ~5M
- Training time: 1 hour
- Quality: Can generate somewhat coherent text
- Use case: Testing, learning, small experiments

### Example 2: Training a Small Model (Single GPU)

**Scenario:** You have a gaming PC and want to train a decent model.

**Your Computer:**
- RTX 3080 (10GB VRAM)
- 16GB RAM
- 5GB of text data (e.g., books, Wikipedia articles)

**Step-by-Step:**

```bash
# 1. Prepare data (takes ~15 minutes)
python data_prep_tool.py \
    --input_dir ./large_corpus \
    --output_dir ./processed_large \
    --vocab_size 32000 \
    --max_length 2048 \
    --num_workers 8

# You'll see:
# Found 1,247 files
# Training bpe tokenizer...
# Created 287,654 training examples

# 2. Train model (takes ~4 days)
python train.py \
    --data_dir ./processed_large \
    --save_dir ./checkpoints \
    --hidden_size 768 \
    --num_layers 12 \
    --num_heads 12 \
    --batch_size 24 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --warmup_steps 2000 \
    --mixed_precision \
    --wandb_project my_llm_experiment

# Monitor progress:
# Step 1000/100000 | Loss: 7.2341 | Tokens/sec: 8432
# Validation Loss: 7.1234
# Step 2000/100000 | Loss: 6.8932 | Tokens/sec: 8521
# ...
```

**Tips for this scenario:**
- Check GPU usage: `nvidia-smi -l 1`
- Should show 90-100% GPU utilization
- If lower, increase batch_size to 32 or 40
- Save checkpoints every 1000 steps (automatic)
- Can pause and resume anytime

**Expected Results:**
- Model parameters: ~125M (GPT-2 Small equivalent)
- Training time: 3-5 days
- Final loss: ~2.5-3.0
- Quality: Can write coherent paragraphs, follow instructions somewhat

### Example 3: Multi-GPU Training (4 GPUs)

**Scenario:** You have a workstation with multiple GPUs or using cloud.

**Your Setup:**
- 4x RTX 3090 (24GB each)
- 64GB RAM
- 50GB of high-quality text data

**Step-by-Step:**

```bash
# 1. Prepare data (takes ~1 hour)
python data_prep_tool.py \
    --input_dir ./massive_corpus \
    --output_dir ./processed_massive \
    --vocab_size 32000 \
    --max_length 2048 \
    --num_workers 16

# You'll see:
# Found 12,847 files
# Created 2,876,540 training examples

# 2. Launch multi-GPU training (takes ~1 week)
torchrun --nproc_per_node=4 train.py \
    --data_dir ./processed_massive \
    --save_dir ./checkpoints_multigpu \
    --hidden_size 1024 \
    --num_layers 24 \
    --num_heads 16 \
    --num_kv_heads 8 \
    --batch_size 256 \
    --micro_batch_size 64 \
    --max_steps 500000 \
    --learning_rate 3e-4 \
    --warmup_steps 5000 \
    --mixed_precision \
    --compile_model \
    --wandb_project big_llm_experiment

# You'll see:
# [Rank 0] Device: cuda:0
# [Rank 1] Device: cuda:1
# [Rank 2] Device: cuda:2
# [Rank 3] Device: cuda:3
# Distributed: True (World size: 4)
# Model parameters: 487,234,560
```

**What's happening:**
```
GPU 0: Processes batch 0-63   ──┐
GPU 1: Processes batch 64-127  ─┤
GPU 2: Processes batch 128-191 ─┼─► Average gradients ──► Update model
GPU 3: Processes batch 192-255 ─┘
```

**Benefits:**
- 4x faster than single GPU (in theory)
- In practice: 3.5x faster (due to communication overhead)
- Can train much larger models

### Example 4: Multi-Node Training (Research Lab Setup)

**Scenario:** Training a large model across multiple machines.

**Your Setup:**
- 4 machines, each with 8x A100 GPUs (80GB)
- High-speed network (InfiniBand)
- 500GB of curated, high-quality data

**Machine Configuration:**

```
Machine 1 (Master): 192.168.1.10
Machine 2:          192.168.1.11
Machine 3:          192.168.1.12
Machine 4:          192.168.1.13
```

**On Machine 1 (Master):**

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=0 \
    --master_addr="192.168.1.10" \
    --master_port=29500 \
    train.py \
    --data_dir /shared/processed_data \
    --save_dir /shared/checkpoints \
    --hidden_size 2048 \
    --num_layers 32 \
    --num_heads 32 \
    --num_kv_heads 8 \
    --batch_size 2048 \
    --micro_batch_size 64 \
    --max_steps 1000000 \
    --learning_rate 2e-4 \
    --warmup_steps 10000 \
    --mixed_precision \
    --compile_model
```

**On Machine 2:**

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=4 \
    --node_rank=1 \
    --master_addr="192.168.1.10" \
    --master_port=29500 \
    train.py [... same parameters ...]
```

**On Machines 3 & 4:** Same but with `--node_rank=2` and `--node_rank=3`

**Expected Results:**
- Model parameters: ~1.5B (competitive with GPT-2 XL)
- Training time: 2-4 weeks
- Final loss: <2.0
- Quality: High-quality text generation, good reasoning

**Visualization of Multi-Node Training:**

```
┌─────────────────────────────────────────────────┐
│ Machine 1 (Master) - 192.168.1.10               │
│ ┌────┬────┬────┬────┬────┬────┬────┬────┐      │
│ │GPU0│GPU1│GPU2│GPU3│GPU4│GPU5│GPU6│GPU7│      │
│ └────┴────┴────┴────┴────┴────┴────┴────┘      │
└─────────────────────────────────────────────────┘
                        │
            ┌───────────┼───────────┐
            │           │           │
┌───────────▼───┐  ┌────▼──────┐  ┌▼──────────┐
│  Machine 2    │  │ Machine 3 │  │ Machine 4 │
│  8 GPUs       │  │ 8 GPUs    │  │ 8 GPUs    │
└───────────────┘  └───────────┘  └───────────┘

Total: 32 GPUs working together
Effective batch size: 2048 examples
Each GPU processes: 64 examples
```

---

## Part 3: Real-World Scenarios

### Scenario A: Training on Custom Domain Data

**Use Case:** You want to train a model that understands medical literature.

**Data Collection:**
```bash
# Your data structure:
medical_texts/
├── textbooks/
│   ├── anatomy.txt
│   ├── physiology.txt
│   └── pharmacology.txt
├── research_papers/
│   ├── paper1.txt
│   ├── paper2.txt
│   └── ...
└── clinical_notes/
    └── (anonymized patient notes)
```

**Preparation:**
```bash
python data_prep_tool.py \
    --input_dir ./medical_texts \
    --output_dir ./medical_processed \
    --tokenizer_type bpe \
    --vocab_size 32000 \
    --max_length 2048
```

**Training:**
```bash
python train.py \
    --data_dir ./medical_processed \
    --hidden_size 768 \
    --num_layers 12 \
    --batch_size 32 \
    --max_steps 100000 \
    --learning_rate 3e-4 \
    --mixed_precision \
    --wandb_project medical_llm
```

**Expected Outcome:**
- Model understands medical terminology
- Can complete medical sentences
- Knows relationships between symptoms and conditions
- Ready for fine-tuning on specific tasks (diagnosis, summarization)

### Scenario B: Continuing Pre-training (Transfer Learning)

**Use Case:** Start from a checkpoint and continue training on new data.

**Step 1: Get a checkpoint**
```bash
# Could be from:
# - Your previous training
# - A pre-trained model
# - A checkpoint shared online
```

**Step 2: Continue training**
```bash
python train.py \
    --data_dir ./new_data \
    --checkpoint ./checkpoints/checkpoint_step_50000.pt \
    --max_steps 150000 \
    --learning_rate 1e-4  # Lower LR for fine-tuning
```

**What happens:**
```
Step 50000: Resume from checkpoint
    ↓
Step 50001: Continue with new data
    ↓
Step 150000: New checkpoint
```

### Scenario C: Experimenting with Hyperparameters

**Goal:** Find the best settings for your use case.

**Experiment 1: Learning Rate**
```bash
# Try different learning rates
python train.py --learning_rate 1e-4 --wandb_run_name lr_1e4
python train.py --learning_rate 3e-4 --wandb_run_name lr_3e4
python train.py --learning_rate 6e-4 --wandb_run_name lr_6e4

# Compare in WandB dashboard
# Best is usually the one with:
# - Smooth loss curve
# - Lowest final validation loss
# - No training instabilities
```

**Experiment 2: Model Size**
```bash
# Small
python train.py --hidden_size 512 --num_layers 8 --wandb_run_name small

# Medium
python train.py --hidden_size 768 --num_layers 12 --wandb_run_name medium

# Large
python train.py --hidden_size 1024 --num_layers 16 --wandb_run_name large

# Compare:
# - Training time
# - Final loss
# - Generation quality
```

---

## Part 4: Understanding What's Happening Inside

### What the Model Learns Over Time

**Steps 0-1,000: Random → Basic Patterns**
```
Step 0:
Input:  "The cat sat on the"
Output: "xzkw qprl mno"  (complete gibberish)

Step 500:
Input:  "The cat sat on the"
Output: "mat dog house tree"  (words, but no sense)

Step 1,000:
Input:  "The cat sat on the"
Output: "mat and then went"  (starting to make sense!)
```

**Steps 1,000-10,000: Grammar and Structure**
```
Step 5,000:
Input:  "The cat sat on the"
Output: "mat and looked around the room"
(grammatically correct!)

Step 10,000:
Input:  "Once upon a time"
Output: "there was a young girl who lived in a small village"
(proper story structure!)
```

**Steps 10,000-50,000: Context and Meaning**
```
Step 25,000:
Input:  "The scientist studied quantum"
Output: "mechanics and made groundbreaking discoveries in particle physics"
(understands domain-specific context!)

Step 50,000:
Input:  "If I heat water to 100°C"
Output: "it will boil and turn into steam, which is the gaseous state of water"
(understands cause and effect!)
```

**Steps 50,000-100,000: Refinement and Coherence**
```
Step 100,000:
Input:  "Explain photosynthesis"
Output: "Photosynthesis is the process by which plants convert sunlight 
         into chemical energy. Chloroplasts in plant cells contain 
         chlorophyll, which absorbs light energy. This energy is used 
         to convert carbon dioxide and water into glucose and oxygen."
(coherent, accurate, well-structured!)
```

### Visualizing Attention Patterns

**Example: Understanding Pronouns**

```
Sentence: "The dog chased the cat because it was hungry"

What word does "it" refer to?

Attention weights when processing "it":
┌─────────┬──────┬────────┬─────────┬─────────┬──────┬─────────┬────────┐
│   The   │  dog │ chased │   the   │   cat   │because│   it   │  was   │
├─────────┼──────┼────────┼─────────┼─────────┼──────┼─────────┼────────┤
│  0.05   │ 0.75 │  0.03  │  0.02   │  0.10   │ 0.02 │  0.02  │  0.01  │
└─────────┴──────┴────────┴─────────┴─────────┴──────┴─────────┴────────┘
          ▲▲▲▲▲
          High attention!

Model correctly identifies "dog" as the referent!
```

### Memory and Computation Requirements

**Quick Reference Table:**

| Model Size | Parameters | Training RAM | GPU Memory | Tokens/sec* | Training Time** |
|-----------|-----------|--------------|------------|-------------|-----------------|
| Tiny      | 20M       | 4GB          | 2GB        | 50,000      | 2 days          |
| Small     | 125M      | 8GB          | 8GB        | 15,000      | 5 days          |
| Medium    | 350M      | 16GB         | 16GB       | 8,000       | 2 weeks         |
| Large     | 800M      | 32GB         | 24GB       | 4,000       | 1 month         |
| XL        | 1.5B      | 64GB         | 40GB       | 2,000       | 2 months        |
| XXL       | 3B        | 128GB        | 80GB       | 1,000       | 3-4 months      |

*On RTX 3090 with mixed precision
**For 100B tokens of training data

### Cost Estimation

**Cloud Training Costs (AWS p3.2xlarge - V100 GPU):**

```
Instance: p3.2xlarge
Cost: $3.06/hour

Small Model (125M params):
- Training time: ~120 hours
- Cost: $367

Medium Model (350M params):
- Training time: ~336 hours  
- Cost: $1,028

Large Model (800M params):
- Training time: ~720 hours
- Cost: $2,203
```

**Cost-Saving Tips:**
1. Use spot instances (70% cheaper, may be interrupted)
2. Train smaller models first to validate approach
3. Use checkpointing - resume if interrupted
4. Consider Lambda Labs or Vast.ai (cheaper alternatives)

---

## Part 5: Common Patterns and Workflows

### Pattern 1: Iterative Improvement

```
Round 1: Quick test
├─ Data: 100MB
├─ Model: Tiny (20M)
├─ Steps: 10,000
└─ Result: Works! Loss decreases.

Round 2: Scale up data
├─ Data: 1GB
├─ Model: Small (125M)
├─ Steps: 50,000
└─ Result: Much better quality!

Round 3: Scale up model
├─ Data: 1GB
├─ Model: Medium (350M)
├─ Steps: 100,000
└─ Result: High quality, ready for use!
```

### Pattern 2: Domain Adaptation

```
Step 1: General pre-training
├─ Data: Wikipedia, books, web text
├─ Model: Medium
└─ Result: General language model

Step 2: Domain-specific training
├─ Data: Medical papers, textbooks
├─ Model: Continue from checkpoint
└─ Result: Medical-specialized model

Step 3: Task-specific fine-tuning
├─ Data: Question-answer pairs
├─ Model: Continue from checkpoint
└─ Result: Medical QA assistant
```

### Pattern 3: Ensemble and Comparison

```
Train multiple models with different settings:

Model A: hidden_size=768, layers=12
Model B: hidden_size=1024, layers=8
Model C: hidden_size=512, layers=16

Compare performance:
├─ Loss curves
├─ Generation quality
├─ Inference speed
└─ Memory usage

Select best for your needs!
```

---

## Part 6: Debugging Guide

### Problem: Loss Not Decreasing

**Symptoms:**
```
Step 100 | Loss: 8.2341
Step 200 | Loss: 8.2356
Step 300 | Loss: 8.2389
Step 400 | Loss: 8.2401
```

**Diagnosis Checklist:**

1. **Learning rate too low?**
```bash
# Try increasing
--learning_rate 1e-3  # Instead of 3e-4
```

2. **Data problem?**
```python
# Check data manually
from datasets import load_from_disk
dataset = load_from_disk("./processed_data/dataset")
print(dataset["train"][0])  # Should show real data, not garbage
```

3. **Model too small?**
```bash
# Try larger model
--hidden_size 1024 --num_layers 16  # Instead of 768/12
```

4. **Batch size too small?**
```bash
# Try larger batch
--batch_size 64  # Instead of 32
```

### Problem: Loss Becomes NaN

**Symptoms:**
```
Step 100 | Loss: 7.2341
Step 200 | Loss: 6.8932
Step 300 | Loss: nan
```

**Solutions:**

1. **Reduce learning rate:**
```bash
--learning_rate 1e-4  # Much lower
```

2. **Increase warmup:**
```bash
--warmup_steps 5000  # Instead of 2000
```

3. **Check for bad data:**
```python
# Look for extremely long sequences or weird characters
import datasets
ds = datasets.load_from_disk("./processed_data/dataset")
lengths = [len(x['input_ids']) for x in ds['train']]
print(f"Max length: {max(lengths)}")
print(f"Min length: {min(lengths)}")
```

4. **Enable gradient clipping:**
```bash
--grad_clip 0.5  # Instead of 1.0 (more aggressive)
```

### Problem: Training Too Slow

**Symptoms:**
```
Tokens/sec: 1,234 (should be 10,000+)
GPU utilization: 30% (should be 90%+)
```

**Solutions:**

1. **Increase batch size:**
```bash
--batch_size 64  # Or as large as memory allows
```

2. **Enable optimizations:**
```bash
--mixed_precision --compile_model
```

3. **Check data loading:**
```python
# In train.py, increase workers:
DataLoader(..., num_workers=8, pin_memory=True)
```

4. **Use faster storage:**
```bash
# Move data to SSD if on HDD
# Or use RAM disk for small datasets
```

---

This visual guide provides concrete examples and diagrams to help you understand exactly what's happening at each stage of training. Use it alongside the code to build intuition about the training process!