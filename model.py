"""
Modern Transformer Architecture with State-of-the-Art Improvements
Includes: RoPE, SwiGLU, RMSNorm, Grouped Query Attention, Flash Attention
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    num_kv_heads: Optional[int] = None  # For Grouped Query Attention (GQA)
    intermediate_size: Optional[int] = None  # For SwiGLU FFN
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.1
    attention_dropout: float = 0.1
    use_flash_attention: bool = True
    tie_word_embeddings: bool = False
    
    def __post_init__(self):
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads  # Standard MHA
        if self.intermediate_size is None:
            self.intermediate_size = 4 * self.hidden_size


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (more efficient than LayerNorm)"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position encoding"""
    
    def __init__(self, dim: int, max_position_embeddings: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
    
    def forward(self, x, seq_len: Optional[int] = None):
        if seq_len is None:
            seq_len = x.shape[1]
        
        # Generate position indices
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        
        return emb.cos()[None, :, :], emb.sin()[None, :, :]


def rotate_half(x):
    """Helper function for applying rotary embeddings"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary position embeddings to queries and keys"""
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention (GQA) - Middle ground between MHA and MQA
    Reduces KV cache size while maintaining quality
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta
        )
        
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.use_flash = config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.size()
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(bsz, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Expand KV heads for grouped attention
        if self.num_kv_groups > 1:
            k = k.repeat_interleave(self.num_kv_groups, dim=1)
            v = v.repeat_interleave(self.num_kv_groups, dim=1)
        
        # Compute attention
        if self.use_flash:
            # Use Flash Attention if available (much faster)
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout.p if self.training else 0.0,
                is_causal=True
            )
        else:
            # Standard attention implementation
            attn_weights = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask
            
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_weights = self.attention_dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        attn_output = self.o_proj(attn_output)
        
        return attn_output


class SwiGLU(nn.Module):
    """
    SwiGLU activation function (Swish-Gated Linear Unit)
    Better than standard FFN with ReLU/GELU
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
    
    def forward(self, x):
        gate = F.silu(self.gate_proj(x))  # Swish activation
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class TransformerBlock(nn.Module):
    """Single transformer layer with modern improvements"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = SwiGLU(config)
        self.attention_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Pre-normalization (more stable than post-norm)
        residual = hidden_states
        hidden_states = self.attention_norm(hidden_states)
        hidden_states = self.attention(hidden_states, attention_mask)
        hidden_states = residual + hidden_states
        
        # FFN with residual
        residual = hidden_states
        hidden_states = self.ffn_norm(hidden_states)
        hidden_states = self.feed_forward(hidden_states)
        hidden_states = residual + hidden_states
        # print(hidden_states.shape)
        return hidden_states
    
 

class LLMModel(nn.Module):
    """Complete Language Model"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
        # Output projection
        if config.tie_word_embeddings:
            self.lm_head = None  # Share weights with embedding
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to causal mask
            batch_size, seq_length = input_ids.shape
            causal_mask = torch.triu(
                torch.ones((seq_length, seq_length), dtype=torch.bool, device=input_ids.device),
                diagonal=1
            )
            attention_mask = attention_mask[:, None, None, :] * (~causal_mask[None, None, :, :])
            attention_mask = attention_mask.masked_fill(attention_mask == 0, float('-inf'))
        
        # Apply transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)
        
        hidden_states = self.norm(hidden_states)
        
        # Compute logits
        if self.config.tie_word_embeddings:
            logits = F.linear(hidden_states, self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1)
            )
        
        return logits, loss
    
    def get_num_params(self, non_embedding=True):
        """Count model parameters"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params



 