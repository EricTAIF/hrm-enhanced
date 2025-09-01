"""
HuggingFace-compatible wrapper for Enhanced HRM with reasoning tokens
- PreTrainedModel interface for Unsloth/TRL compatibility
- Reasoning tokens and chat template
- Latent + token reasoning capabilities
"""

import torch
import torch.nn as nn
import json
from typing import Dict, Optional, List, Tuple, Union
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer, GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast

from model import EnhancedHRM
from modern_tokenizer import ModernReasoningTokenizer


class EnhancedHRMConfig(PretrainedConfig):
    """Configuration for Enhanced HRM model"""
    
    model_type = "enhanced_hrm"
    
    def __init__(
        self,
        vocab_size: int = 32000,
        d_model: int = 512,
        n_heads: int = 8,
        n_planner_layers: int = 4,
        n_worker_layers: int = 6,
        max_cycles: int = 8,
        max_seq_len: int = 1024,
        thought_dim: int = 256,
        dropout: float = 0.1,
        reasoning_tokens: List[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.hidden_size = d_model  # HF compatibility
        self.n_heads = n_heads
        self.num_attention_heads = n_heads  # HF compatibility
        self.n_planner_layers = n_planner_layers
        self.n_worker_layers = n_worker_layers
        self.num_hidden_layers = n_planner_layers + n_worker_layers  # HF compatibility
        self.max_cycles = max_cycles
        self.max_seq_len = max_seq_len
        self.max_position_embeddings = max_seq_len  # HF compatibility
        self.thought_dim = thought_dim
        self.dropout = dropout
        
        # Reasoning tokens for structured thinking
        if reasoning_tokens is None:
            reasoning_tokens = [
                "<think>", "</think>",
                "<plan>", "</plan>", 
                "<work>", "</work>",
                "<SOLUTION>", "</SOLUTION>",
                "<REASONING>", "</REASONING>"
            ]
        self.reasoning_tokens = reasoning_tokens


class EnhancedHRMForCausalLM(PreTrainedModel, GenerationMixin):
    """HuggingFace-compatible Enhanced HRM for causal language modeling"""
    
    config_class = EnhancedHRMConfig
    _tied_weights_keys = ["hrm_core.token_emb.weight", "hrm_core.thought_head.token_head.weight"]
    
    def __init__(self, config: EnhancedHRMConfig):
        super().__init__(config)
        self.config = config
        
        # Core HRM model
        self.hrm_core = EnhancedHRM(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_planner_layers=config.n_planner_layers,
            n_worker_layers=config.n_worker_layers,
            max_cycles=config.max_cycles,
            max_seq_len=config.max_seq_len,
            thought_dim=config.thought_dim,
            dropout=config.dropout
        )
        
        # For GRPO, we need a reference to frozen weights
        self.register_buffer("_is_reference", torch.tensor(False))
        
        # Initialize weights
        self.post_init()
        
    def get_input_embeddings(self):
        return self.hrm_core.token_emb
        
    def set_input_embeddings(self, value):
        self.hrm_core.token_emb = value
        
    def get_output_embeddings(self):
        return self.hrm_core.thought_head.token_head
        
    def set_output_embeddings(self, new_embeddings):
        self.hrm_core.thought_head.token_head = new_embeddings
        
    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings for new reasoning tokens"""
        old_embeddings = self.get_input_embeddings()
        old_output = self.get_output_embeddings()
        
        # Resize input embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        new_embeddings.weight.data[:old_embeddings.num_embeddings] = old_embeddings.weight.data
        
        # Resize output embeddings  
        new_output = nn.Linear(old_output.in_features, new_num_tokens, bias=False)
        new_output.weight.data[:old_output.out_features] = old_output.weight.data
        
        self.set_input_embeddings(new_embeddings)
        self.set_output_embeddings(new_output)
        
        # Update config
        self.config.vocab_size = new_num_tokens
        
        return self.get_input_embeddings()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        max_cycles: Optional[int] = None,
        return_dict: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        max_cycles = max_cycles or self.config.max_cycles
        
        # Forward through HRM with reasoning cycles
        outputs = self.hrm_core(
            input_ids, 
            max_cycles=max_cycles, 
            return_thoughts=True
        )
        
        logits = outputs['logits']
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            # Calculate loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits, shift_labels)
        
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
            
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,  # HRM doesn't use KV cache yet
            hidden_states=None,
            attentions=None
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None, 
        **kwargs
    ):
        """Prepare inputs for generation - HRM-specific"""
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_cycles": kwargs.get("max_cycles", self.config.max_cycles)
        }
    
    def _reorder_cache(self, past_key_values, beam_idx):
        """Reorder cache for beam search (HRM doesn't use cache yet)"""
        return past_key_values


def setup_reasoning_tokenizer(base_model_name: str = "microsoft/Phi-3-mini-4k-instruct") -> AutoTokenizer:
    """Setup tokenizer with reasoning tokens and chat template"""
    
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    
    # Add reasoning tokens
    special_tokens = [
        "<think>", "</think>",
        "<plan>", "</plan>", 
        "<work>", "</work>",
        "<SOLUTION>", "</SOLUTION>",
        "<REASONING>", "</REASONING>",
        "<step>", "</step>",
        "<reflection>", "</reflection>"
    ]
    
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    
    # Reasoning-aware chat template
    chat_template = '''{% for message in messages %}
{%- if message['role'] == 'user' %}
{{ message['content'] }}
{%- elif message['role'] == 'assistant' %}
<think>
Let me think step by step about this problem...
</think>

{{ message['content'] }}
{%- endif %}
{% endfor %}'''
    
    tokenizer.chat_template = chat_template
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return tokenizer


def create_hrm_model_and_tokenizer(
    vocab_size: int = 32019,
    d_model: int = 256,
    n_heads: int = 4,
    max_cycles: int = 6,
    load_checkpoint: Optional[str] = None
) -> Tuple[EnhancedHRMForCausalLM, AutoTokenizer]:
    """Create HRM model and tokenizer for GRPO training"""
    
    # Setup tokenizer
    tokenizer = setup_reasoning_tokenizer()
    actual_vocab_size = len(tokenizer)
    
    # Create config
    config = EnhancedHRMConfig(
        vocab_size=actual_vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        max_cycles=max_cycles
    )
    
    # Create model
    model = EnhancedHRMForCausalLM(config)
    
    # Load checkpoint if provided
    if load_checkpoint and torch.cuda.is_available():
        print(f"🔄 Loading checkpoint: {load_checkpoint}")
        try:
            checkpoint = torch.load(load_checkpoint, map_location='cpu')
            
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Load with prefix mapping for HRM core
            model_dict = {}
            for key, value in state_dict.items():
                if key.startswith('hrm_core.'):
                    # Already has correct prefix
                    model_dict[key] = value
                else:
                    # Add hrm_core prefix
                    model_dict[f'hrm_core.{key}'] = value
                    
            # Load state dict with error handling
            model.load_state_dict(model_dict, strict=False)
            print("✅ Checkpoint loaded successfully")
            
        except Exception as e:
            print(f"⚠️  Warning: Could not load checkpoint: {e}")
            print("📝 Starting with random weights")
    
    return model, tokenizer


if __name__ == "__main__":
    # Test the wrapper
    print("🧪 Testing EnhancedHRM HuggingFace wrapper...")
    
    model, tokenizer = create_hrm_model_and_tokenizer()
    
    print(f"📊 Model vocab size: {model.config.vocab_size}")
    print(f"🔤 Tokenizer vocab size: {len(tokenizer)}")
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Test forward pass
    text = "Let me solve this step by step:"
    inputs = tokenizer(text, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    print(f"✅ Forward pass successful!")
    print(f"📈 Logits shape: {outputs.logits.shape}")
    print(f"🔄 HRM metrics: {getattr(outputs, 'hrm_metrics', 'Not available')}")
    
    # Test generation
    generated = model.generate(
        **inputs,
        max_length=inputs['input_ids'].shape[1] + 20,
        do_sample=True,
        temperature=0.8,
        max_cycles=3  # HRM-specific parameter
    )
    
    generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
    print(f"🎯 Generated: {generated_text}")
    
    print("✨ HRM wrapper ready for GRPO training!")