"""
Training Script for Mixture of Experts LLM
Handles MoE-specific considerations like load balancing and expert utilization
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
 
# Import MoE model
from moe_model import MoELLMModel, MoEConfig
 

@dataclass
class MoETrainingConfig:
    """Training configuration for MoE models"""
    # Model
    model_config: MoEConfig = None
    
    # Data
    data_dir: str = "./processed_data"
    
    # Training
    batch_size: int = 32
    micro_batch_size: Optional[int] = None
    max_steps: int = 100000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    
    # MoE-specific
    router_z_loss_weight: float = 0.001  # Additional regularization for router
    expert_capacity_factor: float = 1.25
    
    # Optimization
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    compile_model: bool = False
    
    # Checkpointing
    save_dir: str = "./checkpoints_moe"
    save_interval: int = 1000
    eval_interval: int = 500
    log_interval: int = 100
    
    # Distributed
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    
    # Logging 
    log_expert_stats: bool = True  # Log expert utilization
    
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


def get_lr(step: int, config: MoETrainingConfig) -> float:
    """Cosine learning rate schedule with warmup"""
    if step < config.warmup_steps:
        return config.learning_rate * step / config.warmup_steps
    
    if step > config.max_steps:
        return config.min_learning_rate
    
    decay_ratio = (step - config.warmup_steps) / (config.max_steps - config.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_learning_rate + coeff * (config.learning_rate - config.min_learning_rate)


class MoETrainer:
    """Main training class for MoE models"""
    
    def __init__(self, config: MoETrainingConfig):
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
        
        # Expert statistics tracking
        self.expert_usage_counts = None
        if config.log_expert_stats:
            self.expert_usage_counts = torch.zeros(
                config.model_config.num_experts,
                device=self.device
            )
         
        if self.is_main_process:
            self._print_model_info()
    
    def _print_model_info(self):
        """Print model information"""
        print(f"\n{'='*60}")
        print(f"MoE Training Configuration")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Distributed: {self.config.distributed} (World size: {self.world_size})")
        print(f"Total parameters: {self.model.get_num_params():,}")
        print(f"Active parameters: {self.model.get_num_active_params():,}")
        
        sparsity = (1 - self.model.get_num_active_params() / self.model.get_num_params()) * 100
        print(f"Sparsity: {sparsity:.1f}%")
        
        print(f"Experts: {self.config.model_config.num_experts}")
        print(f"Active experts per token: {self.config.model_config.num_experts_per_token}")
        print(f"Batch size: {self.config.batch_size} (micro: {self.config.micro_batch_size})")
        print(f"Gradient accumulation steps: {self.config.gradient_accumulation_steps}")
        print(f"Mixed precision: {self.config.mixed_precision}")
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
        model = MoELLMModel(self.config.model_config)
        model = model.to(self.device)
        
        # Compile model for PyTorch 2.0+
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
                # No weight decay for biases, layer norms, and router
                if 'bias' in name or 'norm' in name or 'ln' in name or 'router' in name:
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
        """Single training step with MoE-specific logging"""
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        
        # Forward pass with mixed precision
        with autocast(enabled=(self.scaler is not None)):
            logits, loss, aux_loss = self.model(
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
        
        # Return losses for logging
        lm_loss = loss.item() * self.config.gradient_accumulation_steps
        aux_loss_val = aux_loss.item() if aux_loss is not None else 0.0
        
        return lm_loss, aux_loss_val
    
    @torch.no_grad()
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_lm_loss = 0
        total_aux_loss = 0
        num_batches = 0
        
        for batch in self.val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            with autocast(enabled=(self.scaler is not None)):
                logits, loss, aux_loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=input_ids
                )
            
            total_lm_loss += loss.item()
            if aux_loss is not None:
                total_aux_loss += aux_loss.item()
            num_batches += 1
            
            if num_batches >= 50:  # Limit eval batches
                break
        
        avg_lm_loss = total_lm_loss / num_batches
        avg_aux_loss = total_aux_loss / num_batches if total_aux_loss > 0 else 0.0
        
        # Gather losses from all processes
        if self.config.distributed:
            lm_loss_tensor = torch.tensor(avg_lm_loss, device=self.device)
            aux_loss_tensor = torch.tensor(avg_aux_loss, device=self.device)
            dist.all_reduce(lm_loss_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(aux_loss_tensor, op=dist.ReduceOp.AVG)
            avg_lm_loss = lm_loss_tensor.item()
            avg_aux_loss = aux_loss_tensor.item()
        
        self.model.train()
        return avg_lm_loss, avg_aux_loss
    
    @torch.no_grad()
    def compute_expert_statistics(self):
        """
        Compute expert utilization statistics
        Helps identify if load balancing is working properly
        """
        if not self.config.log_expert_stats:
            return {}
        
        # Get router from first MoE layer
        model = self.model.module if self.config.distributed else self.model
        
        # Sample a few batches to compute statistics
        expert_usage = torch.zeros(
            self.config.model_config.num_experts,
            device=self.device
        )
        
        num_samples = 0
        for i, batch in enumerate(self.train_loader):
            if i >= 10:  # Sample 10 batches
                break
            
            input_ids = batch['input_ids'].to(self.device)
            batch_size, seq_len = input_ids.shape
            
            # Forward pass to get routing decisions
            with autocast(enabled=(self.scaler is not None)):
                logits, _, _ = model(input_ids)
            
            num_samples += batch_size * seq_len
        
        # Compute statistics
        if num_samples > 0:
            expert_usage = expert_usage / num_samples
            
            if self.config.distributed:
                dist.all_reduce(expert_usage, op=dist.ReduceOp.AVG)
            
            stats = {
                'expert_usage_mean': expert_usage.mean().item(),
                'expert_usage_std': expert_usage.std().item(),
                'expert_usage_min': expert_usage.min().item(),
                'expert_usage_max': expert_usage.max().item(),
            }
            
            # Compute entropy (higher = more balanced)
            entropy = -(expert_usage * torch.log(expert_usage + 1e-10)).sum().item()
            stats['expert_entropy'] = entropy
            
            return stats
        
        return {}
    
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
        
        lm_losses = []
        aux_losses = []
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
                lm_loss, aux_loss = self.train_step(batch)
                lm_losses.append(lm_loss)
                aux_losses.append(aux_loss)
                
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
                        avg_lm_loss = sum(lm_losses) / len(lm_losses)
                        avg_aux_loss = sum(aux_losses) / len(aux_losses)
                        elapsed = time.time() - start_time
                        tokens_per_sec = (self.config.batch_size * 
                                        self.config.model_config.max_position_embeddings * 
                                        self.config.log_interval) / elapsed
                        
                        print(f"Step {self.step}/{self.config.max_steps} | "
                              f"LM Loss: {avg_lm_loss:.4f} | "
                              f"Aux Loss: {avg_aux_loss:.6f} | "
                              f"LR: {lr:.2e} | "
                              f"Tokens/sec: {tokens_per_sec:.0f}")
             
                        
                        lm_losses = []
                        aux_losses = []
                        start_time = time.time()
                    
                    # Evaluation
                    if self.step % self.config.eval_interval == 0:
                        val_lm_loss, val_aux_loss = self.evaluate()
                        
                        if self.is_main_process:
                            print(f"Validation - LM Loss: {val_lm_loss:.4f}, Aux Loss: {val_aux_loss:.6f}")
                             
                            # Compute expert statistics
                            if self.config.log_expert_stats and self.step % (self.config.eval_interval * 2) == 0:
                                expert_stats = self.compute_expert_statistics()
                                if expert_stats:
                                    print(f"Expert Statistics:")
                                    for key, val in expert_stats.items():
                                        print(f"  {key}: {val:.4f}")
                      
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
    parser = argparse.ArgumentParser(description="Train MoE LLM")
    
    # Model args
    parser.add_argument("--vocab_size", type=int, default=32000)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--num_kv_heads", type=int, default=None)
    
    # MoE-specific args
    parser.add_argument("--num_experts", type=int, default=8, help="Number of experts")
    parser.add_argument("--num_experts_per_token", type=int, default=2, help="Top-k experts to activate")
    parser.add_argument("--aux_loss_weight", type=float, default=0.01, help="Load balancing loss weight")
    parser.add_argument("--moe_layers", type=str, default=None, help="Comma-separated layer indices for MoE (e.g., '2,4,6,8')")
    
    # Training args
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, default="./checkpoints_moe")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # System args
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--compile_model", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    
    # Logging 
    parser.add_argument("--log_expert_stats", action="store_true", help="Log expert utilization statistics")
    
    args = parser.parse_args()
    
    # Parse MoE layers if provided
    moe_layers = None
    if args.moe_layers:
        moe_layers = [int(x) for x in args.moe_layers.split(',')]
    
    # Create configs
    model_config = MoEConfig(
        vocab_size=args.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        num_experts=args.num_experts,
        num_experts_per_token=args.num_experts_per_token,
        aux_loss_weight=args.aux_loss_weight,
        moe_layers=moe_layers,
    )
    
    train_config = MoETrainingConfig(
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
        log_expert_stats=args.log_expert_stats,
    )
    
    # Initialize trainer
    trainer = MoETrainer(train_config)
    
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
