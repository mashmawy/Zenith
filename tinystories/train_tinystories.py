"""
train_tinystories.py
Train a small language model on TinyStories dataset with:
- Checkpoint every 100 steps
- Automatic GPU detection
- Resume capability
- Auto-cleanup of old checkpoints
"""
import sys
import os

# Get the absolute path of the directory that contains the 'model_architecture.py' file. 
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir) # This steps up from 'tinystories' to 'llm'

# Add the root directory to the system path
sys.path.append(root_dir)
import torch 
from torch.utils.data import DataLoader
from datasets import load_from_disk 
import os
import json
import math
import time
import glob 

 
# Import your model
from model import LLMModel, ModelConfig


class TinyStoriesTrainer:
    """Trainer for TinyStories with robust checkpointing"""
    
    def __init__(
        self,
        data_dir="./tinystories_processed", 
        output_dir="./tinystories_model",
        hidden_size=384,
        num_layers=6,
        num_heads=6,
        batch_size=64,
        learning_rate=3e-4,
        max_steps=50000,
        warmup_steps=1000,
        mixed_precision=True, 
        log_interval=100,
        eval_interval=1000,
        save_interval=100,
        keep_last_n_checkpoints=10,
    ):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.mixed_precision = mixed_precision
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        self.keep_last_n_checkpoints = keep_last_n_checkpoints
        
        # Force GPU if available
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"\n✅ Using GPU: {torch.cuda.get_device_name(0)}")
            print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
        else:
            self.device = torch.device("cpu")
            print("\n⚠️  WARNING: No GPU detected! Training will be VERY slow on CPU.")
            response = input("Continue on CPU? (yes/no): ")
            if response.lower() != 'yes':
                print("Exiting. Please enable GPU for training.")
                exit(1)
        
        # Load metadata
        with open(f"{data_dir}/metadata.json", 'r') as f:
            metadata = json.load(f)
        
        # Model config
        self.model_config = ModelConfig(
            vocab_size=metadata['vocab_size'],
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            max_position_embeddings=metadata['max_length'],
        )
        
        # Training config
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps
        
        # Initialize training state
        self.step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        # Setup
        self.setup_data()
        self.setup_model()
        self.setup_optimizer()
        
        if mixed_precision and self.device.type == "cuda":
            self.scaler = torch.cuda.amp.GradScaler("cuda")
        else:
            self.scaler = None
         
        
        # Print configuration
        print("="*60)
        print("TinyStories Training Configuration")
        print("="*60)
        print(f"Device: {self.device}")
        print(f"Model parameters: {self.count_parameters():,}")
        print(f"Training examples: {len(self.train_dataset):,}")
        print(f"Validation examples: {len(self.val_dataset):,}")
        print(f"Batch size: {batch_size}")
        print(f"Max steps: {max_steps}")
        print(f"Save interval: every {save_interval} steps")
        print(f"Mixed precision: {mixed_precision}")
        print("="*60 + "\n")
    
    def setup_data(self):
        """Load tokenized dataset"""
        print("Loading tokenized dataset...")
        dataset = load_from_disk(self.data_dir)
        
        self.train_dataset = dataset['train']
        self.val_dataset = dataset['validation']
        
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    def setup_model(self):
        """Initialize model"""
        print("Initializing model...")
        self.model = LLMModel(self.model_config)
        self.model = self.model.to(self.device)
        self.model.train()
    
    def setup_optimizer(self):
        """Setup AdamW optimizer"""
        decay_params = []
        nodecay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'bias' in name or 'norm' in name:
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)
        
        self.optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': 0.1},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ], lr=self.learning_rate, betas=(0.9, 0.95))
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def get_lr(self, step):
        """Cosine learning rate schedule with warmup"""
        if step < self.warmup_steps:
            return self.learning_rate * step / self.warmup_steps
        
        if step > self.max_steps:
            return self.learning_rate * 0.1
        
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.learning_rate * 0.1 + 0.9 * self.learning_rate * (1 + math.cos(math.pi * progress)) / 2
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = torch.stack(batch['input_ids']).to(self.device,non_blocking=True)
        attention_mask = torch.stack(batch['attention_mask']).to(self.device).to(torch.float32,non_blocking=True)
        
        # Forward pass
        if self.scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, loss = self.model(input_ids, attention_mask, labels=input_ids)
        else:
            logits, loss = self.model(input_ids, attention_mask, labels=input_ids)
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        self.optimizer.zero_grad(set_to_none=True)
        
        return loss.item()
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            if num_batches >= 50:
                break
            
            input_ids = torch.stack(batch['input_ids']).to(self.device,non_blocking=True)
            attention_mask = torch.stack(batch['attention_mask']).to(self.device).to(torch.float32,non_blocking=True)
                
            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits, loss = self.model(input_ids, attention_mask, labels=input_ids)
            else:
                logits, loss = self.model(input_ids, attention_mask, labels=input_ids)
            
            total_loss += loss.item()
            num_batches += 1
        
        self.model.train()
        return total_loss / num_batches
    
    def save_checkpoint(self, is_best=False):
        """Save checkpoint with automatic cleanup"""
        os.makedirs(self.output_dir, exist_ok=True)
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'model_config': {
                'vocab_size': self.model_config.vocab_size,
                'hidden_size': self.model_config.hidden_size,
                'num_layers': self.model_config.num_layers,
                'num_heads': self.model_config.num_heads,
                'max_position_embeddings': self.model_config.max_position_embeddings,
            }
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        # Save numbered checkpoint
        checkpoint_path = f"{self.output_dir}/checkpoint_step_{self.step}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save as latest
        latest_path = f"{self.output_dir}/checkpoint_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save as best if applicable
        if is_best:
            best_path = f"{self.output_dir}/checkpoint_best.pt"
            torch.save(checkpoint, best_path)
            print(f"⭐ New best model saved! (step {self.step})")
        else:
            print(f"💾 Checkpoint saved: step {self.step}")
        
        # Cleanup old checkpoints
        self.cleanup_old_checkpoints()
    
    def cleanup_old_checkpoints(self):
        """Remove old checkpoints, keeping only last N"""
        checkpoints = glob.glob(f"{self.output_dir}/checkpoint_step_*.pt")
        
        if len(checkpoints) <= self.keep_last_n_checkpoints:
            return
        
        # Sort by step number
        checkpoints.sort(key=lambda x: int(x.split('_step_')[1].split('.pt')[0]))
        
        # Remove oldest
        to_remove = checkpoints[:-self.keep_last_n_checkpoints]
        for checkpoint in to_remove:
            os.remove(checkpoint)
    
    def load_checkpoint(self, checkpoint_path):
        """Load checkpoint and resume training"""
        print(f"\n📂 Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.step = checkpoint['step']
        self.epoch = checkpoint.get('epoch', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✅ Resumed from step {self.step}")
        print(f"   Best validation loss: {self.best_val_loss:.4f}\n")
    
    def find_latest_checkpoint(self):
        """Find latest checkpoint"""
        latest_path = f"{self.output_dir}/checkpoint_latest.pt"
        
        if os.path.exists(latest_path):
            return latest_path
        
        checkpoints = glob.glob(f"{self.output_dir}/checkpoint_step_*.pt")
        if not checkpoints:
            return None
        
        checkpoints.sort(key=lambda x: int(x.split('_step_')[1].split('.pt')[0]))
        return checkpoints[-1]
    
    def train(self):
        """Main training loop"""
        print("Starting training...\n")
        
        losses = []
        start_time = time.time()
        
        while self.step < self.max_steps:
            for batch in self.train_loader:
                # Update learning rate
                lr = self.get_lr(self.step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Training step
                loss = self.train_step(batch)
                losses.append(loss)
                self.step += 1
                
                # Logging
                if self.step % self.log_interval == 0:
                    avg_loss = sum(losses) / len(losses)
                    elapsed = time.time() - start_time
                    tokens_per_sec = (self.batch_size * 512 * self.log_interval) / elapsed
                    
                    print(f"Step {self.step:6d}/{self.max_steps} | "
                          f"Loss: {avg_loss:.4f} | "
                          f"LR: {lr:.2e} | "
                          f"Tokens/s: {tokens_per_sec:,.0f}")
          
                    losses = []
                    start_time = time.time()
                
                # Evaluation
                if self.step % self.eval_interval == 0:
                    val_loss = self.evaluate()
                    
                    is_best = val_loss < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_loss
                    
                    print(f"📊 Validation loss: {val_loss:.4f} (best: {self.best_val_loss:.4f})")
 
                
                # Save checkpoint
                if self.step % self.save_interval == 0:
                    self.save_checkpoint(is_best=False)
                
                if self.step >= self.max_steps:
                    break
            
            self.epoch += 1
        
        # Final save
        self.save_checkpoint()
        print("\n" + "="*60)
        print("✅ Training complete!")
        print(f"   Total steps: {self.step}")
        print(f"   Best validation loss: {self.best_val_loss:.4f}")
        print("="*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train on TinyStories dataset")
    
    # Paths
    parser.add_argument("--data_dir", type=str, default="./tinystories_processed")
    parser.add_argument("--output_dir", type=str, default="./tinystories_model")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path")
    
    # Model
    parser.add_argument("--hidden_size", type=int, default=384)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--warmup_steps", type=int, default=1000)
       
    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    
    # System
    parser.add_argument("--no_mixed_precision", action="store_true") 
  
    # Checkpointing
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--keep_checkpoints", type=int, default=10)
    
    args = parser.parse_args()
    
    # GPU warning
    if not torch.cuda.is_available():
        print("\n" + "="*60)
        print("⚠️  WARNING: CUDA not available!")
        print("="*60)
        print("GPU training is strongly recommended.")
        print("CPU training will be 50-100x slower.\n")
    
    # Create trainer
    trainer = TinyStoriesTrainer(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        mixed_precision=not args.no_mixed_precision, 
        save_interval=args.save_interval,
        keep_last_n_checkpoints=args.keep_checkpoints,
        num_heads=args.num_heads,
        warmup_steps=args.warmup_steps
    )
    
    # Resume if requested
    if args.resume or args.checkpoint:
        if args.checkpoint:
            trainer.load_checkpoint(args.checkpoint)
        else:
            latest = trainer.find_latest_checkpoint()
            if latest:
                trainer.load_checkpoint(latest)
            else:
                print("⚠️  No checkpoint found. Starting fresh...\n")
    
    # Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted!")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("✅ Saved. Resume with: python train_tinystories.py --resume") 
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Saving checkpoint...")
        trainer.save_checkpoint() 
        raise


if __name__ == "__main__":
    main()
