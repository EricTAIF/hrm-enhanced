"""
Enhanced Hierarchical Reasoning Model with Latent Space Thinking
- Dual thought channels (explicit + latent)
- Dynamic cycle gating based on entropy
- Cross-cycle consistency losses
- RoPE + cycle-aware normalization
- State skip connections for deep reasoning
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class LatentThought:
    """Container for latent reasoning states"""
    planner_state: torch.Tensor      # High-level planning thoughts
    worker_state: torch.Tensor       # Low-level execution thoughts  
    thought_logits: torch.Tensor     # Internal reasoning logits
    confidence: torch.Tensor         # Confidence in current reasoning


class RotaryEmbedding(nn.Module):
    """RoPE for both token-time and cycle-time"""
    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.max_seq_len = max_seq_len
        
        # Precompute frequencies
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        self.register_buffer("freqs", freqs, persistent=False)
        
    def forward(self, seq_len: int, cycle_idx: int = 0):
        # Token positions
        t = torch.arange(seq_len, device=self.freqs.device, dtype=self.freqs.dtype)
        # Add cycle offset for cycle-aware positioning
        t = t + cycle_idx * seq_len * 0.1  # Small offset per cycle
        
        freqs = torch.outer(t, self.freqs)
        cos_vals = torch.cos(freqs)
        sin_vals = torch.sin(freqs)
        return cos_vals, sin_vals


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary position embedding"""
    # x: [batch, seq_len, n_heads, head_dim]
    # cos, sin: [seq_len, head_dim//2]
    
    B, T, H, D = x.shape
    
    # Ensure cos/sin have correct dimensions
    if cos.shape[0] != T:
        cos = cos[:T]
        sin = sin[:T]
    
    # Ensure the feature dimension matches
    if cos.shape[1] * 2 != D:
        # Adjust dimensions by repeating or truncating
        target_dim = D // 2
        if cos.shape[1] < target_dim:
            # Repeat to match
            repeat_factor = target_dim // cos.shape[1]
            cos = cos.repeat(1, repeat_factor)[:, :target_dim]
            sin = sin.repeat(1, repeat_factor)[:, :target_dim]
        else:
            # Truncate
            cos = cos[:, :target_dim]
            sin = sin[:, :target_dim]
    
    # Split features into pairs
    x1 = x[..., 0::2]  # Even indices
    x2 = x[..., 1::2]  # Odd indices
    
    # Expand cos/sin to match batch and head dimensions
    cos_expanded = cos.view(1, T, 1, -1).expand(B, T, H, -1)
    sin_expanded = sin.view(1, T, 1, -1).expand(B, T, H, -1)
    
    # Apply rotation
    rotated_x1 = x1 * cos_expanded - x2 * sin_expanded
    rotated_x2 = x1 * sin_expanded + x2 * cos_expanded
    
    # Interleave back
    output = torch.zeros_like(x)
    output[..., 0::2] = rotated_x1
    output[..., 1::2] = rotated_x2
    
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, 
                attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        q = apply_rope(q.transpose(1, 2), cos, sin).transpose(1, 2)
        k = apply_rope(k.transpose(1, 2), cos, sin).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU activation from PaLM"""
    def __init__(self, d_model: int, expansion_factor: float = 2.75):
        super().__init__()
        hidden_dim = int(d_model * expansion_factor)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class ReasoningBlock(nn.Module):
    """Enhanced reasoning block with cycle-aware normalization"""
    def __init__(self, d_model: int, n_heads: int, expansion_factor: float = 2.75):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.mlp = SwiGLU(d_model, expansion_factor)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        # Cycle embedding for cycle-aware processing
        self.cycle_emb = nn.Embedding(32, d_model)  # Support up to 32 cycles
        
    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, 
                cycle_idx: int, skip_connection: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Add cycle information
        cycle_bias = self.cycle_emb(torch.tensor(cycle_idx, device=x.device))
        x = x + cycle_bias.unsqueeze(0).unsqueeze(0)
        
        # Skip connection from earlier cycles
        if skip_connection is not None:
            x = x + 0.1 * skip_connection  # Small skip weight
            
        # Self-attention with RoPE
        attn_out = self.attn(self.norm1(x), cos, sin)
        x = x + attn_out
        
        # MLP
        mlp_out = self.mlp(self.norm2(x))
        x = x + mlp_out
        
        return x


class ThoughtChannelHead(nn.Module):
    """Dual head for explicit tokens + latent thoughts"""
    def __init__(self, d_model: int, vocab_size: int, thought_dim: int = 256):
        super().__init__()
        self.vocab_size = vocab_size
        self.thought_dim = thought_dim
        
        # Next token prediction
        self.token_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Latent thought projection  
        self.thought_proj = nn.Linear(d_model, thought_dim)
        self.thought_norm = RMSNorm(thought_dim)
        
        # Confidence estimation
        self.confidence_head = nn.Linear(d_model, 1)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Token logits for next token prediction
        token_logits = self.token_head(x)
        
        # Latent thought representation
        thought_repr = self.thought_norm(self.thought_proj(x))
        
        # Confidence in reasoning
        confidence = torch.sigmoid(self.confidence_head(x))
        
        return token_logits, thought_repr, confidence


class CycleGatingModule(nn.Module):
    """Dynamic cycle gating based on entropy and confidence"""
    def __init__(self, d_model: int):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, 1)
        self.entropy_threshold = nn.Parameter(torch.tensor(2.0))
        
    def should_continue_reasoning(self, hidden_state: torch.Tensor, 
                                token_logits: torch.Tensor, confidence: torch.Tensor) -> torch.Tensor:
        # Compute entropy of token distribution
        probs = F.softmax(token_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Gate signal from hidden state
        gate_logit = self.gate_proj(hidden_state.mean(dim=1))  # [B, 1]
        
        # Continue reasoning if:
        # 1. High entropy (uncertain about next token)  
        # 2. Low confidence in current reasoning
        # 3. Gate signal is positive
        high_entropy = entropy.mean(dim=-1, keepdim=True) > self.entropy_threshold
        low_confidence = confidence.mean(dim=-1) < 0.7
        gate_positive = torch.sigmoid(gate_logit) > 0.5
        
        should_continue = (high_entropy | low_confidence) & gate_positive
        return should_continue.squeeze(-1)


class EnhancedHRM(nn.Module):
    """Enhanced Hierarchical Reasoning Model with Latent Space Thinking"""
    
    def __init__(self, vocab_size: int = 32000, d_model: int = 512, n_heads: int = 8,
                 n_planner_layers: int = 4, n_worker_layers: int = 6,
                 max_cycles: int = 8, max_seq_len: int = 1024,
                 thought_dim: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_cycles = max_cycles
        self.max_seq_len = max_seq_len
        self.thought_dim = thought_dim
        
        # Token + position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # RoPE for cycle-aware positioning
        self.rope = RotaryEmbedding(d_model // n_heads, max_seq_len)
        
        # Hierarchical reasoning modules
        self.planner_layers = nn.ModuleList([
            ReasoningBlock(d_model, n_heads) for _ in range(n_planner_layers)
        ])
        
        self.worker_layers = nn.ModuleList([
            ReasoningBlock(d_model, n_heads) for _ in range(n_worker_layers)
        ])
        
        # Initial states (learnable)
        self.init_planner_state = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.init_worker_state = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        
        # Thought channel heads
        self.thought_head = ThoughtChannelHead(d_model, vocab_size, thought_dim)
        
        # Dynamic cycle gating
        self.cycle_gate = CycleGatingModule(d_model)
        
        # Cross-cycle consistency
        self.cycle_consistency_proj = nn.Linear(d_model, d_model)
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def embed_inputs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed input tokens with position encoding"""
        B, T = input_ids.shape
        
        # Token embeddings
        token_embs = self.token_emb(input_ids)
        
        # Position embeddings  
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        pos_embs = self.pos_emb(pos_ids)
        
        # Combine and scale
        embeddings = (token_embs + pos_embs) * math.sqrt(self.d_model)
        return self.dropout(embeddings)
        
    def reasoning_cycle(self, planner_state: torch.Tensor, worker_state: torch.Tensor,
                       input_embs: torch.Tensor, cycle_idx: int,
                       skip_states: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, LatentThought]:
        """Single cycle of hierarchical reasoning"""
        
        B, T, C = input_embs.shape
        cos, sin = self.rope(T, cycle_idx)
        
        # Skip connection from previous cycles  
        skip_connection = skip_states[-2] if len(skip_states) >= 2 else None
        
        # High-level planning phase
        planner_input = planner_state + input_embs  # Inject current input
        for layer in self.planner_layers:
            planner_state = layer(planner_state, cos, sin, cycle_idx, skip_connection)
            
        # Low-level working phase - informed by planner
        worker_input = worker_state + planner_state  # Planning guides execution
        for layer in self.worker_layers:
            worker_state = layer(worker_state, cos, sin, cycle_idx, skip_connection)
        
        # Generate latent thoughts
        thought_repr = self.cycle_consistency_proj(worker_state)
        token_logits, latent_thought, confidence = self.thought_head(worker_state)
        
        latent_thoughts = LatentThought(
            planner_state=planner_state.detach(),
            worker_state=worker_state.detach(),  
            thought_logits=latent_thought,
            confidence=confidence
        )
        
        return planner_state, worker_state, latent_thoughts
        
    def forward(self, input_ids: torch.Tensor, max_cycles: Optional[int] = None,
                return_thoughts: bool = False) -> Dict[str, torch.Tensor]:
        """Forward pass with dynamic multi-cycle reasoning"""
        
        B, T = input_ids.shape
        device = input_ids.device
        max_cycles = max_cycles or self.max_cycles
        
        # Input embeddings
        input_embs = self.embed_inputs(input_ids)
        
        # Initialize reasoning states
        planner_state = self.init_planner_state.expand(B, T, -1)
        worker_state = self.init_worker_state.expand(B, T, -1)
        
        # Track states for skip connections and consistency
        all_token_logits = []
        all_thoughts = []
        skip_states = []
        
        # Multi-cycle reasoning loop
        for cycle in range(max_cycles):
            planner_state, worker_state, thoughts = self.reasoning_cycle(
                planner_state, worker_state, input_embs, cycle, skip_states
            )
            
            # Store for skip connections
            skip_states.append(worker_state.clone())
            
            # Generate outputs
            token_logits, _, confidence = self.thought_head(worker_state)
            all_token_logits.append(token_logits)
            all_thoughts.append(thoughts)
            
            # Dynamic gating - should we continue reasoning?
            if cycle < max_cycles - 1:  # Don't gate on last cycle
                should_continue = self.cycle_gate.should_continue_reasoning(
                    worker_state, token_logits, confidence
                )
                
                # If most sequences don't need more reasoning, stop
                if should_continue.float().mean() < 0.3:
                    break
        
        # Final outputs
        final_logits = all_token_logits[-1]
        
        # Prepare return dict
        outputs = {
            'logits': final_logits,
            'cycles_used': torch.tensor(len(all_token_logits), device=device),
        }
        
        if return_thoughts:
            outputs['thoughts'] = all_thoughts
            outputs['all_logits'] = torch.stack(all_token_logits, dim=1)  # [B, cycles, T, vocab]
            
        # Cross-cycle consistency loss (for training)
        if len(all_token_logits) > 1:
            mid_logits = all_token_logits[len(all_token_logits)//2]
            consistency_loss = F.kl_div(
                F.log_softmax(mid_logits, dim=-1),
                F.softmax(final_logits.detach(), dim=-1),
                reduction='batchmean'
            )
            outputs['consistency_loss'] = consistency_loss
            
        return outputs


if __name__ == "__main__":
    # Test the enhanced HRM
    model = EnhancedHRM(
        vocab_size=1000, 
        d_model=256,
        n_heads=4,
        max_cycles=6
    )
    
    # Test input
    input_ids = torch.randint(0, 1000, (2, 64))
    
    print("🧠 Enhanced HRM with Latent Space Reasoning!")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Forward pass
    outputs = model(input_ids, return_thoughts=True)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Cycles used: {outputs['cycles_used']}")
    print(f"Has consistency loss: {'consistency_loss' in outputs}")
    print("✨ LATENT SPACE REASONING ACTIVATED! ✨")