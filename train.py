"""
Scalable LLM Training Framework
Supports: Single CPU/GPU, Multi-GPU (DDP), Multi-Node training
Includes: Mixed precision, gradient accumulation, checkpointing 
"""

import os
import json
import math
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from datasets import load_from_disk
from tqdm import tqdm

# Import model
from model_architecture import LLMModel, ModelConfig
 

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model
    model_config: ModelConfig = None
    
    # Data
    data_dir: str = "./processed_data"
    
    # Training
    batch_size: int = 32
    micro_batch_size: Optional[int] = None  # For gradient accumulation
    max_steps: int = 100000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # Optimization
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = False  # torch.compile for PyTorch 2.0+
    
    # Checkpointing
    save_dir: str = "./checkpoints"
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 100
    
    # Distributed
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
     
    
    def __post_init__(self):
        if self.micro_batch_size is None:
            self.micro_batch_size = self.batch_size
        self.gradient_accumulation_steps = self.batch_size // self.micro_batch_size


def setup_distributed():
    """Initialize distributed training"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_lr(step: int, config: TrainingConfig) -> float:
    """Cosine learning rate schedule with warmup"""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    if step > config.max_steps:
        return config.min_learning_rate
    
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


class Trainer:
    """Main training class"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.step = 0
        self.epoch = 0
        
        # Setup distributed
        self.rank, self.world_size, self.local_rank = setup_distributed()
        self.config.rank = self.rank
        self.config.world_size = self.world_size
        self.config.local_rank = self.local_rank
        self.config.distributed = self.world_size > 1
        
        self.is_main_process = self.rank == 0
        
        # Setup device
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.device = torch.device("cpu")
        
        # Load data
        self.train_loader, self.val_loader = self.setup_data()
        
        # Setup model
        self.model = self.setup_model()
        
        # Setup optimizer
        self.optimizer = self.setup_optimizer()
        self.scaler = GradScaler() if config.mixed_precision and torch.cuda.is_available() else None
     
        if self.is_main_process:
            print(f"\n{'='*60}")
            print(f"Training Configuration")
            print(f"{'='*60}")
            print(f"Device: {self.device}")
            print(f"Distributed: {self.config.distributed} (World size: {self.world_size})")
            print(f"Model parameters: {self.model.get_num_params():,}")
            print(f"Batch size: {config.batch_size} (micro: {config.micro_batch_size})")
            print(f"Gradient accumulation steps: {config.gradient_accumulation_steps}")
            print(f"Mixed precision: {config.mixed_precision}")
            print(f"{'='*60}\n")
    
    def setup_data(self):
        """Load and setup data loaders"""
        dataset = load_from_disk(self.config.data_dir + "/dataset")
        train_dataset = dataset["train"]
        val_dataset = dataset["test"]
        
        # Setup samplers for distributed training
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        ) if self.config.distributed else None
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=False
        ) if self.config.distributed else None
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.micro_batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=4,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.micro_batch_size,
            sampler=val_sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def setup_model(self):
        """Initialize and setup model"""
        model = LLMModel(self.config.model_config)
        model = model.to(self.device)
        
        # Compile model for PyTorch 2.0+ (faster training)
        if self.config.compile_model and hasattr(torch, 'compile'):
            if self.is_main_process:
                print("Compiling model with torch.compile()...")
            model = torch.compile(model)
        
        # Wrap with DDP for distributed training
        if self.config.distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        return model
    
    def setup_optimizer(self):
        """Setup AdamW optimizer with weight decay"""
        # Separate parameters for weight decay
        decay_params = []
        nodecay_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # No weight decay for biases and layer norms
                if 'bias' in name or 'norm' in name or 'ln' in name:
                    nodecay_params.append(param)
                else:
                    decay_params.append(param)
        
        optimizer = torch.optim.AdamW([
            {'params': decay_params, 'weight_decay': self.config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps
        )
        
        return optimizer
    
    def train_step(self, batch):
        """Single training step"""
        input_ids = torch.stack(batch['input_ids']).to(self.device,non_blocking=True) 
        attention_mask = torch.stack(batch['attention_mask']).to(self.device).to(torch.float32,non_blocking=True)
 
             
        # Forward pass with mixed precision
        with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=(self.scaler is not None)):
            logits, loss = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids
            )
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.config.gradient_accumulation_steps
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = torch.stack(batch['input_ids']).to(self.device,non_blocking=True) 
            attention_mask = torch.stack(batch['attention_mask']).to(self.device).to(torch.float32,non_blocking=True)
        
            
            with autocast('cuda' if torch.cuda.is_available() else 'cpu', enabled=(self.scaler is not None)):
                logits, loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= 50:  # Limit eval batches for speed
                break
        
        avg_loss = total_loss / num_batches
        
        # Gather losses from all processes
        if self.config.distributed:
            loss_tensor = torch.tensor(avg_loss, device=self.device)
            dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
            avg_loss = loss_tensor.item()
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, filename: str = None):
        """Save training checkpoint"""
        if not self.is_main_process:
            return
        
        os.makedirs(self.config.save_dir, exist_ok=True)
        
        if filename is None:
            filename = f"checkpoint_step_{self.step}.pt"
        
        # Get model state dict (unwrap DDP if needed)
        model_state = self.model.module.state_dict() if self.config.distributed else self.model.state_dict()
        
        checkpoint = {
            'step': self.step,
            'epoch': self.epoch,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': asdict(self.config),
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        save_path = Path(self.config.save_dir) / filename
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Load model state
        if self.config.distributed:
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load training state
        self.step = checkpoint['step']
        self.epoch = checkpoint['epoch']
        
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        if self.is_main_process:
            print(f"Checkpoint loaded from step {self.step}")
    
    def train(self):
        """Main training loop"""
        self.model.train()
        
        train_losses = []
        start_time = time.time()
        
        while self.step < self.config.max_steps:
            if self.config.distributed:
                self.train_loader.sampler.set_epoch(self.epoch)
            
            for batch_idx, batch in enumerate(self.train_loader):
                # Update learning rate
                lr = get_lr(self.step, self.config)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Training step
                loss = self.train_step(batch)
                train_losses.append(loss)
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    # Gradient clipping
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                        self.optimizer.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                    
                    # Logging
                    if self.step % self.config.log_interval == 0 and self.is_main_process:
                        avg_loss = sum(train_losses) / len(train_losses)
                        elapsed = time.time() - start_time
                        tokens_per_sec = (self.config.batch_size * self.config.model_config.max_position_embeddings * 
                                        self.config.log_interval) / elapsed
                        
                        print(f"Step {self.step}/{self.config.max_steps} | "
                              f"Loss: {avg_loss:.4f} | "
                              f"LR: {lr:.2e} | "
                              f"Tokens/sec: {tokens_per_sec:.0f}")
                         
                        train_losses = []
                        start_time = time.time()
                    
                    # Evaluation
                    if self.step % self.config.eval_interval == 0:
                        val_loss = self.evaluate()
                        if self.is_main_process:
                            print(f"Validation Loss: {val_loss:.4f}")
 
                    # Checkpointing
                    if self.step % self.config.save_interval == 0:
                        self.save_checkpoint()
                    
                    # Check if training complete
                    if self.step >= self.config.max_steps:
                        break
            
            self.epoch += 1
            
            if self.step >= self.config.max_steps:
                break
        
        # Save final checkpoint
        if self.is_main_process:
            self.save_checkpoint("final_checkpoint.pt")
            print("\n" + "="*60)
            print("Training completed!")
            print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Train LLM")
    
    # Model args
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=2)
    parser.add_argument("--num_kv_heads", type=int, default=None)
    
    # Training args
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # System args
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
     
    
    args = parser.parse_args()
    
    # Create configs
    model_config = ModelConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
    )
    
    train_config = TrainingConfig(
        model_config=model_config,
        data_dir=args.data_dir,
        save_dir=args.save_dir,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        mixed_precision=args.mixed_precision,
        compile_model=args.compile_model, 
    )
    
    # Initialize trainer
    trainer = Trainer(train_config)
    
    # Load checkpoint if provided
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    try:
        trainer.train()
    finally:
        cleanup_distributed() 


if __name__ == "__main__":
    main()
