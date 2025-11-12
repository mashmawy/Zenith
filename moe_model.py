"""
Mixture of Experts (MoE) Architecture for LLM Training
Implements sparse MoE with top-k routing and load balancing
"""
 
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple

# Import base components from original model
from model import (
    RMSNorm,  
    GroupedQueryAttention
)


@dataclass
class MoEConfig:
    """Configuration for Mixture of Experts model"""
    # Standard model parameters
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: Optional[int] = None
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_flash_attention: bool = True
    tie_word_embeddings: bool = False
    
    # MoE-specific parameters
    num_experts: int = 8  # Total number of expert networks
    num_experts_per_token: int = 2  # Top-k experts to activate
    expert_capacity_factor: float = 1.25  # For load balancing
    aux_loss_weight: float = 0.01  # Weight for load balancing loss
    
    # Expert FFN size (if None, uses standard 4*hidden_size)
    expert_intermediate_size: Optional[int] = None
    
    # Which layers use MoE (None = all layers)
    moe_layers: Optional[list] = None  # e.g., [2, 4, 6, 8, 10] for sparse MoE
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.expert_intermediate_size is None:
            self.expert_intermediate_size = 4 * self.hidden_size


class Expert(nn.Module):
    """
    Single expert network (SwiGLU FFN)
    Each expert is a specialist in processing certain types of information
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.expert_intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.expert_intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch * seq_len, hidden_size]
        Returns:
            output: [batch * seq_len, hidden_size]
        """
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class Router(nn.Module):
    """
    Router network that decides which experts to use for each token
    Uses learned routing weights to compute expert scores
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        
        # Router weights: project hidden state to expert scores
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            expert_indices: [batch * seq_len, num_experts_per_token] - which experts to use
            expert_weights: [batch * seq_len, num_experts_per_token] - weights for each expert
            router_logits: [batch * seq_len, num_experts] - raw routing scores (for aux loss)
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten batch and sequence dimensions
        hidden_states = hidden_states.view(-1, hidden_size)  # [batch * seq_len, hidden_size]
        
        # Compute routing logits
        router_logits = self.gate(hidden_states)  # [batch * seq_len, num_experts]
        
        # Select top-k experts
        routing_weights = F.softmax(router_logits, dim=-1)
        expert_weights, expert_indices = torch.topk(
            routing_weights, 
            self.num_experts_per_token, 
            dim=-1
        )
        
        # Normalize weights (so they sum to 1 for each token)
        expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)
        
        return expert_indices, expert_weights, router_logits


class MixtureOfExperts(nn.Module):
    """
    Sparse Mixture of Experts layer
    Routes each token to top-k experts and combines their outputs
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.hidden_size = config.hidden_size
        
        # Create router
        self.router = Router(config)
        
        # Create expert networks
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
    
    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through MoE layer
        
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            
        Returns:
            output: [batch, seq_len, hidden_size]
            router_logits: [batch * seq_len, num_experts] - for computing aux loss
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Flatten for routing
        hidden_states_flat = hidden_states.view(-1, hidden_size)  # [batch * seq_len, hidden_size]
        
        # Route tokens to experts
        expert_indices, expert_weights, router_logits = self.router(hidden_states)
        # expert_indices: [batch * seq_len, num_experts_per_token]
        # expert_weights: [batch * seq_len, num_experts_per_token]
        
        # Initialize output
        output = torch.zeros_like(hidden_states_flat)
        
        # Process tokens through selected experts
        # For efficiency, we group tokens by expert
        for expert_idx in range(self.num_experts):
            # Find which tokens use this expert
            token_expert_mask = (expert_indices == expert_idx).any(dim=-1)
            
            if not token_expert_mask.any():
                continue  # No tokens for this expert
            
            # Get tokens for this expert
            expert_input = hidden_states_flat[token_expert_mask]
            
            # Process through expert
            expert_output = self.experts[expert_idx](expert_input)
            
            # Get weights for this expert
            # Find positions where this expert is selected
            expert_positions = (expert_indices == expert_idx)
            weights = torch.zeros(hidden_states_flat.size(0), device=hidden_states.device)
            
            for k in range(self.num_experts_per_token):
                mask = expert_positions[:, k]
                weights[mask] = expert_weights[:, k][mask]
            
            # Add weighted expert output
            output[token_expert_mask] += expert_output * weights[token_expert_mask].unsqueeze(-1)
        
        # Reshape back to original
        output = output.view(batch_size, seq_len, hidden_size)
        
        return output, router_logits


class MoETransformerBlock(nn.Module):
    """
    Transformer block with Mixture of Experts instead of standard FFN
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        
        # Attention (same as standard transformer)
        self.attention = GroupedQueryAttention(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # MoE instead of standard FFN
        self.moe = MixtureOfExperts(config)
        self.moe_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with MoE
        
        Returns:
            hidden_states: Output tensor
            router_logits: For computing auxiliary loss
        """
        # Attention with residual
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # MoE with residual
        residual = hidden_states
        hidden_states = self.moe_norm(hidden_states)
        hidden_states, router_logits = self.moe(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states, router_logits


def compute_load_balancing_loss(router_logits: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    Compute auxiliary loss to encourage balanced expert usage
    
    This prevents the model from always using the same experts and
    ensures all experts are utilized during training.
    
    Args:
        router_logits: [num_tokens, num_experts]
        num_experts: Total number of experts
        
    Returns:
        loss: Scalar load balancing loss
    """
    # Compute fraction of tokens routed to each expert
    routing_weights = F.softmax(router_logits, dim=-1)
    expert_usage = routing_weights.mean(dim=0)  # [num_experts]
    
    # Compute how many tokens each expert would get in perfect balance
    target_usage = 1.0 / num_experts
    
    # L2 loss between actual and target usage
    # Penalizes deviation from uniform distribution
    load_loss = torch.sum((expert_usage - target_usage) ** 2)
    
    # Scale by number of experts
    load_loss = load_loss * num_experts
    
    return load_loss


class MoELLMModel(nn.Module):
    """
    Complete Language Model with Mixture of Experts
    Similar to models like GPT-4, Mixtral, Switch Transformer
    """
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        
        # Determine which layers use MoE
        if config.moe_layers is None:
            # All layers use MoE
            self.moe_layer_indices = set(range(config.num_layers))
        else:
            self.moe_layer_indices = set(config.moe_layers)
        
        # Create transformer layers (mix of standard and MoE)
        self.layers = nn.ModuleList()
        for i in range(config.num_layers):
            if i in self.moe_layer_indices:
                self.layers.append(MoETransformerBlock(config))
            else:
                # Import standard transformer block from original model
                from model import TransformerBlock
                self.layers.append(TransformerBlock(config))
        
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output projection
        if config.tie_word_embeddings:
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass
        
        Returns:
            logits: [batch, seq_len, vocab_size]
            loss: Language modeling loss (if labels provided)
            aux_loss: Load balancing auxiliary loss
        """
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            batch_size, seq_length = input_ids.shape
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device),
                diagonal=1
            )
            attention_mask = attention_mask[:, None, None, :] * (~causal_mask[None, None, :, :])
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        
        # Collect router logits for load balancing loss
        all_router_logits = []
        
        # Pass through transformer layers
        for i, layer in enumerate(self.layers):
            if i in self.moe_layer_indices:
                # MoE layer - returns router logits
                hidden_states, router_logits = layer(hidden_states, attention_mask)
                all_router_logits.append(router_logits)
            else:
                # Standard layer
                hidden_states = layer(hidden_states, attention_mask)
        
        # Final normalization
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute losses
        lm_loss = None
        aux_loss = None
        
        if labels is not None:
            # Language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        # Load balancing auxiliary loss
        if len(all_router_logits) > 0:
            # Concatenate all router logits
            router_logits = torch.cat(all_router_logits, dim=0)
            aux_loss = compute_load_balancing_loss(router_logits, self.config.num_experts)
            aux_loss = aux_loss * self.config.aux_loss_weight
        
        # Combine losses
        total_loss = None
        if lm_loss is not None:
            total_loss = lm_loss
            if aux_loss is not None:
                total_loss = total_loss + aux_loss
        
        return logits, total_loss, aux_loss
    
    def get_num_params(self, non_embedding=True):
        """Count model parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
    
    def get_num_active_params(self):
        """
        Count parameters that are active for a single forward pass
        This is less than total params because only top-k experts are used
        """
        # Embedding + attention + norms
        active_params = self.embed_tokens.weight.numel()
        
        for i, layer in enumerate(self.layers):
            if i in self.moe_layer_indices:
                # Attention params
                active_params += sum(p.numel() for p in layer.attention.parameters())
                active_params += sum(p.numel() for p in layer.attention_norm.parameters())
                active_params += sum(p.numel() for p in layer.moe_norm.parameters())
                
                # Router params
                active_params += sum(p.numel() for p in layer.moe.router.parameters())
                
                # Only k experts (not all experts)
                expert_params = sum(p.numel() for p in layer.moe.experts[0].parameters())
                active_params += expert_params * self.config.num_experts_per_token
            else:
                # Standard layer - all params active
                active_params += sum(p.numel() for p in layer.parameters())
        
        # Output head
        active_params += self.norm.weight.numel()
        if not self.config.tie_word_embeddings:
            active_params += self.lm_head.weight.numel()
        
        return active_params


def calculate_moe_model_size(config: MoEConfig):
    """
    Calculate and display MoE model statistics
    """
    print("\n" + "="*60)
    print("MoE MODEL SIZE CALCULATOR")
    print("="*60)
    
    print("\n📐 Architecture:")
    print(f"  Hidden size: {config.hidden_size:,}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Attention heads: {config.num_heads}")
    print(f"  Experts per layer: {config.num_experts}")
    print(f"  Active experts per token: {config.num_experts_per_token}")
    
    # Calculate params
    model = MoELLMModel(config)
    total_params = model.get_num_params()
    active_params = model.get_num_active_params()
    
    print(f"\n🔢 Parameters:")
    print(f"  Total: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Active per forward pass: {active_params:,} ({active_params/1e9:.2f}B)")
    print(f"  Sparsity: {(1 - active_params/total_params)*100:.1f}% of params inactive")
    
    print(f"\n💡 MoE Benefits:")
    print(f"  • {config.num_experts}x more parameters than dense model")
    print(f"  • But only {config.num_experts_per_token}/{config.num_experts} experts active")
    print(f"  • ~{config.num_experts/config.num_experts_per_token:.1f}x capacity increase")
    print(f"  • Similar compute cost to {active_params/1e9:.1f}B dense model")
    
    print("="*60 + "\n")
