"""
Utility Scripts for LLM Training Framework
Helpful tools for data inspection, model testing, and monitoring
"""

# ============================================================================
# 1. DATA INSPECTION TOOL
# ============================================================================

def inspect_prepared_data(data_dir: str):
    """
    Inspect prepared dataset to verify it looks correct
    
    Args:
        data_dir: Path to processed data directory
    
    Example:
        python -c "from utils import inspect_prepared_data; inspect_prepared_data('./processed_data')"
    """
    import json
    from datasets import load_from_disk
    from tokenizers import Tokenizer
    
    print("\n" + "="*60)
    print("DATA INSPECTION REPORT")
    print("="*60)
    
    # Load metadata
    with open(f"{data_dir}/metadata.json", 'r') as f:
        metadata = json.load(f)
    
    print("\n📊 Dataset Statistics:")
    print(f"  Total examples: {metadata['num_examples']:,}")
    print(f"  Train examples: {metadata['train_examples']:,}")
    print(f"  Validation examples: {metadata['val_examples']:,}")
    print(f"  Max sequence length: {metadata['max_length']}")
    print(f"  Vocabulary size: {metadata['vocab_size']:,}")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file(f"{data_dir}/tokenizer.json")
    print("\n🔤 Tokenizer Info:")
    print(f"  Vocab size: {tokenizer.get_vocab_size():,}")
    
    # Load dataset
    dataset = load_from_disk(f"{data_dir}/dataset")
    
    print("\n📝 Sample Training Examples:")
    for i in range(min(3, len(dataset['train']))):
        example = dataset['train'][i]
        tokens = example['input_ids'][:50]  # First 50 tokens
        decoded = tokenizer.decode(tokens)
        print(f"\n  Example {i+1}:")
        print(f"    Length: {len(example['input_ids'])} tokens")
        print(f"    Preview: {decoded[:200]}...")
    
    # Analyze length distribution
    lengths = [len(ex['input_ids']) for ex in dataset['train'].select(range(min(1000, len(dataset['train']))))]
    print("\n📏 Sequence Length Distribution (sample of 1000):")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Average: {sum(lengths)/len(lengths):.1f}")
    print(f"  Median: {sorted(lengths)[len(lengths)//2]}")
    
    print("\n✅ Data inspection complete!")
    print("="*60 + "\n")


# ============================================================================
# 2. MODEL SIZE CALCULATOR
# ============================================================================

def calculate_model_size(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    num_kv_heads: int = None,
    intermediate_size: int = None
):
    """
    Calculate model parameters and memory requirements
    
    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        intermediate_size: FFN intermediate size
    
    Example:
        from utils import calculate_model_size
        calculate_model_size(hidden_size=768, num_layers=12)
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads
    if intermediate_size is None:
        intermediate_size = 4 * hidden_size
    
    head_dim = hidden_size // num_heads
    
    print("\n" + "="*60)
    print("MODEL SIZE CALCULATOR")
    print("="*60)
    
    print("\n📐 Architecture:")
    print(f"  Hidden size: {hidden_size:,}")
    print(f"  Layers: {num_layers}")
    print(f"  Attention heads: {num_heads}")
    print(f"  KV heads: {num_kv_heads} {'(GQA)' if num_kv_heads < num_heads else ''}")
    print(f"  Intermediate size: {intermediate_size:,}")
    print(f"  Vocabulary: {vocab_size:,}")
    
    # Calculate parameters
    embedding_params = vocab_size * hidden_size
    
    # Per layer
    attention_params = (
        hidden_size * hidden_size +  # Q projection
        hidden_size * (num_kv_heads * head_dim) +  # K projection
        hidden_size * (num_kv_heads * head_dim) +  # V projection
        hidden_size * hidden_size  # Output projection
    )
    
    ffn_params = (
        hidden_size * intermediate_size +  # Gate projection
        hidden_size * intermediate_size +  # Up projection
        intermediate_size * hidden_size  # Down projection
    )
    
    norm_params = hidden_size * 2  # Two RMSNorm per layer
    
    layer_params = attention_params + ffn_params + norm_params
    total_layer_params = layer_params * num_layers
    
    final_norm_params = hidden_size
    lm_head_params = vocab_size * hidden_size  # Output projection
    
    total_params = embedding_params + total_layer_params + final_norm_params + lm_head_params
    
    print("\n🔢 Parameters:")
    print(f"  Embedding: {embedding_params:,} ({embedding_params/1e6:.1f}M)")
    print(f"  Transformer layers: {total_layer_params:,} ({total_layer_params/1e6:.1f}M)")
    print(f"  Output head: {lm_head_params:,} ({lm_head_params/1e6:.1f}M)")
    print(f"  Total: {total_params:,} ({total_params/1e6:.1f}M or {total_params/1e9:.2f}B)")
    
    # Memory estimates (FP32)
    fp32_memory = total_params * 4 / (1024**3)  # 4 bytes per param, convert to GB
    fp16_memory = total_params * 2 / (1024**3)  # 2 bytes per param
    
    # Training memory (rough estimate: model + gradients + optimizer states)
    training_multiplier = 4  # Model + gradients + 2x optimizer states (Adam)
    fp32_training = fp32_memory * training_multiplier
    fp16_training = fp16_memory * training_multiplier * 0.6  # Mixed precision uses less
    
    print("\n💾 Memory Requirements:")
    print(f"  Model (FP32): {fp32_memory:.2f} GB")
    print(f"  Model (FP16): {fp16_memory:.2f} GB")
    print(f"  Training (FP32): {fp32_training:.2f} GB")
    print(f"  Training (FP16/Mixed): {fp16_training:.2f} GB")
    
    print("\n💡 Recommendations:")
    if fp16_training < 8:
        print("  ✅ Can train on consumer GPU (RTX 3070+)")
    elif fp16_training < 24:
        print("  ✅ Can train on high-end GPU (RTX 3090, RTX 4090)")
    elif fp16_training < 80:
        print("  ⚠️  Requires professional GPU (A100, H100)")
    else:
        print("  ⚠️  Requires multiple professional GPUs")
    
    print("="*60 + "\n")
    
    return total_params


# ============================================================================
# 3. TRAINING MONITOR
# ============================================================================

def monitor_training(checkpoint_dir: str, watch: bool = True):
    """
    Monitor training progress from checkpoints
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        watch: If True, continuously update (like 'watch' command)
    
    Example:
        python -c "from utils import monitor_training; monitor_training('./checkpoints')"
    """
    import time
    import os
    from pathlib import Path
    
    def get_latest_checkpoint(checkpoint_dir):
        checkpoints = list(Path(checkpoint_dir).glob("checkpoint_step_*.pt"))
        if not checkpoints:
            return None
        latest = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
        return latest
    
    def display_status():
        os.system('clear' if os.name == 'posix' else 'cls')
        print("\n" + "="*60)
        print("TRAINING MONITOR")
        print("="*60)
        
        checkpoint = get_latest_checkpoint(checkpoint_dir)
        
        if checkpoint is None:
            print("\n❌ No checkpoints found yet. Training may not have started.")
            return
        
        import torch
        ckpt = torch.load(checkpoint, map_location='cpu')
        
        step = ckpt['step']
        
        print(f"\n📍 Latest Checkpoint: {checkpoint.name}")
        print(f"   Step: {step:,}")
        
        if 'epoch' in ckpt:
            print(f"   Epoch: {ckpt['epoch']}")
        
        # Estimate progress if max_steps is in config
        if 'config' in ckpt and 'max_steps' in ckpt['config']:
            max_steps = ckpt['config']['max_steps']
            progress = (step / max_steps) * 100
            print(f"   Progress: {progress:.1f}% ({step:,}/{max_steps:,})")
            
            # Estimate time remaining
            checkpoint_times = []
            for cp in sorted(Path(checkpoint_dir).glob("checkpoint_step_*.pt")):
                checkpoint_times.append((
                    int(cp.stem.split('_')[-1]),
                    cp.stat().st_mtime
                ))
            
            if len(checkpoint_times) >= 2:
                steps_diff = checkpoint_times[-1][0] - checkpoint_times[-2][0]
                time_diff = checkpoint_times[-1][1] - checkpoint_times[-2][1]
                steps_per_sec = steps_diff / time_diff if time_diff > 0 else 0
                
                remaining_steps = max_steps - step
                remaining_time = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                hours = int(remaining_time // 3600)
                minutes = int((remaining_time % 3600) // 60)
                
                print(f"   Est. time remaining: {hours}h {minutes}m")
        
        # List all checkpoints
        all_checkpoints = sorted(Path(checkpoint_dir).glob("checkpoint_step_*.pt"))
        print(f"\n📁 Total Checkpoints: {len(all_checkpoints)}")
        print("   Recent checkpoints:")
        for cp in all_checkpoints[-5:]:
            size_mb = cp.stat().st_size / (1024**2)
            mod_time = time.ctime(cp.stat().st_mtime)
            print(f"     {cp.name}: {size_mb:.1f} MB, {mod_time}")
        
        print("\n" + "="*60)
        
        if watch:
            print("Refreshing every 30 seconds... (Ctrl+C to stop)")
    
    if watch:
        try:
            while True:
                display_status()
                time.sleep(30)
        except KeyboardInterrupt:
            print("\n\nMonitoring stopped.")
    else:
        display_status()


# ============================================================================
# 4. CHECKPOINT MANAGER
# ============================================================================

def manage_checkpoints(checkpoint_dir: str, keep: int = 5, keep_every: int = None):
    """
    Clean up old checkpoints, keeping only recent ones
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep: Number of most recent checkpoints to keep
        keep_every: Additionally keep every Nth checkpoint (e.g., every 10000 steps)
    
    Example:
        from utils import manage_checkpoints
        manage_checkpoints('./checkpoints', keep=5, keep_every=10000)
    """
    from pathlib import Path
    import os
    
    checkpoints = sorted(
        Path(checkpoint_dir).glob("checkpoint_step_*.pt"),
        key=lambda p: int(p.stem.split('_')[-1])
    )
    
    if len(checkpoints) <= keep:
        print(f"✅ Only {len(checkpoints)} checkpoints, nothing to delete.")
        return
    
    to_keep = set()
    
    # Keep most recent
    to_keep.update(checkpoints[-keep:])
    
    # Keep every Nth
    if keep_every:
        for cp in checkpoints:
            step = int(cp.stem.split('_')[-1])
            if step % keep_every == 0:
                to_keep.add(cp)
    
    # Keep final checkpoint
    if (Path(checkpoint_dir) / "final_checkpoint.pt").exists():
        to_keep.add(Path(checkpoint_dir) / "final_checkpoint.pt")
    
    # Delete others
    deleted = 0
    freed_space = 0
    
    for cp in checkpoints:
        if cp not in to_keep:
            size = cp.stat().st_size
            os.remove(cp)
            deleted += 1
            freed_space += size
            print(f"🗑️  Deleted: {cp.name}")
    
    print(f"\n✅ Cleanup complete:")
    print(f"   Deleted: {deleted} checkpoints")
    print(f"   Freed: {freed_space / (1024**3):.2f} GB")
    print(f"   Remaining: {len(to_keep)} checkpoints")


# ============================================================================
# 5. QUICK MODEL TEST
# ============================================================================

def test_model(checkpoint_path: str, prompt: str = "Once upon a time"):
    """
    Quick test of trained model with text generation
    
    Args:
        checkpoint_path: Path to checkpoint file
        prompt: Text prompt to continue
    
    Example:
        from utils import test_model
        test_model('./checkpoints/checkpoint_step_50000.pt', 'The quick brown fox')
    """
    import torch
    from tokenizers import Tokenizer
    from model_architecture import LLMModel, ModelConfig
    
    print("\n" + "="*60)
    print("MODEL TESTING")
    print("="*60)
    
    # Load checkpoint
    print("\n📦 Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Get config from checkpoint
    config_dict = checkpoint['config']
    model_config = ModelConfig(
        vocab_size=config_dict['model_config']['vocab_size'],
        hidden_size=config_dict['model_config']['hidden_size'],
        num_layers=config_dict['model_config']['num_layers'],
        num_heads=config_dict['model_config']['num_heads'],
    )
    
    # Load model
    print("🧠 Loading model...")
    model = LLMModel(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Load tokenizer
    data_dir = config_dict['data_dir']
    tokenizer = Tokenizer.from_file(f"{data_dir}/tokenizer.json")
    
    print(f"💬 Prompt: {prompt}")
    print("\n📝 Generated text:")
    print("-" * 60)
    
    # Tokenize
    tokens = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([tokens]).to(device)
    
    # Generate
    max_new_tokens = 100
    generated = list(tokens)
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Forward pass
            logits, _ = model(input_ids)
            
            # Get next token (greedy)
            next_token = logits[0, -1].argmax().item()
            generated.append(next_token)
            
            # Update input
            input_ids = torch.tensor([generated]).to(device)
            
            # Stop at end token if exists
            if next_token == 2:  # Common EOS token
                break
    
    # Decode
    output_text = tokenizer.decode(generated)
    print(output_text)
    print("-" * 60)
    print("="*60 + "\n")


# ============================================================================
# 6. DATASET STATISTICS
# ============================================================================

def analyze_dataset_statistics(data_dir: str):
    """
    Detailed statistical analysis of dataset
    
    Args:
        data_dir: Path to processed data directory
    
    Example:
        from utils import analyze_dataset_statistics
        analyze_dataset_statistics('./processed_data')
    """
    from datasets import load_from_disk
    from collections import Counter
    import numpy as np
    
    print("\n" + "="*60)
    print("DETAILED DATASET STATISTICS")
    print("="*60)
    
    dataset = load_from_disk(f"{data_dir}/dataset")
    train_data = dataset['train']
    
    # Length statistics
    lengths = [len(ex['input_ids']) for ex in train_data]
    
    print("\n📊 Sequence Length Analysis:")
    print(f"  Count: {len(lengths):,}")
    print(f"  Min: {min(lengths)}")
    print(f"  Max: {max(lengths)}")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Median: {np.median(lengths):.2f}")
    print(f"  Std Dev: {np.std(lengths):.2f}")
    print(f"  25th percentile: {np.percentile(lengths, 25):.0f}")
    print(f"  75th percentile: {np.percentile(lengths, 75):.0f}")
    print(f"  95th percentile: {np.percentile(lengths, 95):.0f}")
    
    # Token frequency
    print("\n🔤 Token Frequency Analysis:")
    all_tokens = []
    sample_size = min(10000, len(train_data))
    for ex in train_data.select(range(sample_size)):
        all_tokens.extend(ex['input_ids'])
    
    token_counts = Counter(all_tokens)
    print(f"  Unique tokens: {len(token_counts):,}")
    print(f"  Total tokens (sample): {len(all_tokens):,}")
    print(f"  Most common tokens: {token_counts.most_common(10)}")
    
    # Data distribution
    print("\n📈 Length Distribution:")
    bins = [0, 256, 512, 1024, 2048, 4096]
    for i in range(len(bins)-1):
        count = sum(1 for l in lengths if bins[i] <= l < bins[i+1])
        pct = (count / len(lengths)) * 100
        print(f"  {bins[i]}-{bins[i+1]}: {count:,} ({pct:.1f}%)")
    
    print("\n" + "="*60 + "\n")


# ============================================================================
# 7. QUICK SETUP SCRIPT
# ============================================================================

def quick_setup():
    """
    Interactive setup wizard for beginners
    
    Example:
        python -c "from utils import quick_setup; quick_setup()"
    """
    print("\n" + "="*60)
    print("LLM TRAINING FRAMEWORK - QUICK SETUP")
    print("="*60)
    
    print("\n👋 Welcome! Let's set up your training environment.\n")
    
    # Check Python version
    import sys
    print(f"✓ Python version: {sys.version.split()[0]}")
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch installed: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠  CUDA not available (CPU only)")
    except ImportError:
        print("❌ PyTorch not installed!")
        print("   Install with: pip install torch")
        return
    
    # Check other dependencies
    deps = ['tokenizers', 'datasets', 'tqdm']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✓ {dep} installed")
        except ImportError:
            print(f"❌ {dep} not installed")
            print(f"   Install with: pip install {dep}")
     
    
    print("\n" + "="*60)
    print("\n📁 Recommended folder structure:")
    print("""
    my_llm_project/
    ├── data_prep_tool.py
    ├── model.py
    ├── train.py
    ├── utils.py (this file)
    ├── requirements.txt
    ├── raw_text_data/      (put your .txt files here)
    ├── processed_data/     (will be created)
    └── checkpoints/        (will be created)
    """)
    
    print("\n🚀 Next steps:")
    print("1. Put your text files in raw_text_data/")
    print("2. Run: python data_prep_tool.py --input_dir ./raw_text_data --output_dir ./processed_data")
    print("3. Run: python train.py --data_dir ./processed_data --hidden_size 768 --num_layers 12")
    print("\n" + "="*60 + "\n")


# ============================================================================
# MAIN - Run utilities from command line
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="LLM Training Utilities")
    parser.add_argument("command", choices=[
        "inspect", "calculate", "monitor", "cleanup", "test", "analyze", "setup"
    ])
    parser.add_argument("--data_dir", type=str, help="Data directory")
    parser.add_argument("--checkpoint_dir", type=str, help="Checkpoint directory")
    parser.add_argument("--checkpoint", type=str, help="Checkpoint file")
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--keep", type=int, default=5)
    parser.add_argument("--prompt", type=str, default="Once upon a time")
    
    args = parser.parse_args()
    
    if args.command == "inspect":
        inspect_prepared_data(args.data_dir)
    
    elif args.command == "calculate":
        calculate_model_size(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            num_heads=args.num_heads
        )
    
    elif args.command == "monitor":
        monitor_training(args.checkpoint_dir, watch=True)
    
    elif args.command == "cleanup":
        manage_checkpoints(args.checkpoint_dir, keep=args.keep)
    
    elif args.command == "test":
        test_model(args.checkpoint, prompt=args.prompt)
    
    elif args.command == "analyze":
        analyze_dataset_statistics(args.data_dir)
    
    elif args.command == "setup":
        quick_setup()
