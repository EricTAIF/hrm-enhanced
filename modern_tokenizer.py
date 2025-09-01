"""
Modern tokenizer using Phi-3 tokenizer for Enhanced HRM
Much faster and more efficient than training from scratch
"""

from transformers import AutoTokenizer
import torch
from typing import List, Dict


class ModernReasoningTokenizer:
    """Wrapper around Phi-3 tokenizer with reasoning tokens"""
    
    def __init__(self, model_name: str = "microsoft/Phi-3-mini-4k-instruct"):
        print(f"🔤 Loading {model_name} tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        
        # Add special reasoning tokens
        special_tokens = [
            "<think>", "</think>",     # Thinking wrapper
            "<plan>", "</plan>",       # Planning phase  
            "<work>", "</work>",       # Working phase
            "<reason>", "</reason>",   # Reasoning chain
        ]
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        
        # Get vocab info
        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        
        print(f"✅ Tokenizer ready! Vocab size: {self.vocab_size}")
        
    def encode(self, text: str, add_reasoning_tokens: bool = False) -> List[int]:
        """Encode text to token IDs"""
        if add_reasoning_tokens:
            # Add reasoning structure
            text = f"<think> <plan> {text} </plan> <work> [reasoning process] </work> </think>"
        
        return self.tokenizer.encode(text, add_special_tokens=False)
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs back to text"""
        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def batch_encode(self, texts: List[str], max_length: int = 512, 
                    padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        """Batch encode texts"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
    
    def get_special_token_ids(self) -> Dict[str, int]:
        """Get special token IDs"""
        special_tokens = {
            "think_start": self.tokenizer.convert_tokens_to_ids("<think>"),
            "think_end": self.tokenizer.convert_tokens_to_ids("</think>"),
            "plan_start": self.tokenizer.convert_tokens_to_ids("<plan>"),
            "plan_end": self.tokenizer.convert_tokens_to_ids("</plan>"),
            "work_start": self.tokenizer.convert_tokens_to_ids("<work>"),
            "work_end": self.tokenizer.convert_tokens_to_ids("</work>"),
            "pad": self.pad_token_id,
            "eos": self.tokenizer.eos_token_id,
        }
        return special_tokens


if __name__ == "__main__":
    # Test the modern tokenizer
    tokenizer = ModernReasoningTokenizer()
    
    # Test encoding/decoding
    test_text = "The model should learn to reason step by step through complex problems."
    
    print(f"\n🧪 Testing tokenizer:")
    print(f"Original: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded[:20]}... (length: {len(encoded)})")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Test with reasoning tokens
    encoded_reasoning = tokenizer.encode(test_text, add_reasoning_tokens=True)
    decoded_reasoning = tokenizer.decode(encoded_reasoning, skip_special_tokens=False)
    print(f"With reasoning: {decoded_reasoning}")
    
    # Test batch encoding
    texts = [
        "This is a test sentence.",
        "Let me think about this problem step by step.",
        "The reasoning process involves multiple cycles."
    ]
    
    batch = tokenizer.batch_encode(texts, max_length=128)
    print(f"\nBatch encoding shapes:")
    for key, value in batch.items():
        print(f"  {key}: {value.shape}")
        
    print(f"Special token IDs: {tokenizer.get_special_token_ids()}")
    print("✅ Modern tokenizer working perfectly!")