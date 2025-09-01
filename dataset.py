"""
Enhanced dataset for training HRM with latent reasoning
- Supports reasoning-aware batching
- Dynamic sequence length handling
- Thought-channel supervision
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from tokenizer import LatentReasoningTokenizer


class ReasoningDataset(Dataset):
    """Dataset that encourages multi-step reasoning"""
    
    def __init__(self, texts: List[str], tokenizer: LatentReasoningTokenizer,
                 max_seq_len: int = 512, reasoning_prob: float = 0.3):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reasoning_prob = reasoning_prob
        
        # Pre-tokenize all texts
        print("🔄 Pre-tokenizing dataset...")
        self.tokenized_texts = []
        
        for text in texts:
            # Randomly add reasoning structure during training
            add_reasoning = random.random() < reasoning_prob
            tokens = tokenizer.encode(text, add_reasoning_tokens=add_reasoning)
            
            if len(tokens) > 10:  # Filter very short sequences
                self.tokenized_texts.append(tokens)
                
        print(f"✅ Dataset ready with {len(self.tokenized_texts)} samples")
        
    def __len__(self):
        return len(self.tokenized_texts)
        
    def __getitem__(self, idx):
        tokens = self.tokenized_texts[idx]
        
        # Truncate or pad to max length
        if len(tokens) > self.max_seq_len:
            # Random crop for variety
            start_idx = random.randint(0, len(tokens) - self.max_seq_len)
            tokens = tokens[start_idx:start_idx + self.max_seq_len]
        else:
            # Pad with pad tokens
            tokens = tokens + [self.tokenizer.vocab['<pad>']] * (self.max_seq_len - len(tokens))
            
        # Convert to tensors
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)  # Input sequence
        labels = torch.tensor(tokens[1:], dtype=torch.long)      # Shifted for next-token prediction
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = (input_ids != self.tokenizer.vocab['<pad>']).float()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


class ThoughtSupervisionDataset(Dataset):
    """Dataset with explicit thought supervision for better reasoning"""
    
    def __init__(self, texts: List[str], tokenizer: LatentReasoningTokenizer,
                 max_seq_len: int = 512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        
        # Create reasoning examples
        self.examples = self._create_reasoning_examples()
        
    def _create_reasoning_examples(self) -> List[Dict]:
        """Create examples with explicit reasoning structure"""
        examples = []
        
        reasoning_templates = [
            "Let me think step by step. {text}",
            "To understand this: {text}",  
            "Breaking this down: {text}",
            "The reasoning is: {text}",
            "Step by step analysis: {text}"
        ]
        
        for text in self.texts[:1000]:  # Use subset for explicit supervision
            if len(text.split()) < 5:
                continue
                
            # Create reasoning version
            template = random.choice(reasoning_templates)
            reasoning_text = template.format(text=text)
            
            # Tokenize with reasoning structure
            full_text = f"<think> <plan> {reasoning_text} </plan> <work> {text} </work> </think>"
            tokens = self.tokenizer.encode(full_text)
            
            if 10 < len(tokens) <= self.max_seq_len:
                examples.append({
                    'tokens': tokens,
                    'has_reasoning': True,
                    'original_text': text
                })
                
        print(f"📝 Created {len(examples)} reasoning examples")
        return examples
        
    def __len__(self):
        return len(self.examples)
        
    def __getitem__(self, idx):
        example = self.examples[idx]
        tokens = example['tokens']
        
        # Pad if necessary
        if len(tokens) < self.max_seq_len:
            tokens = tokens + [self.tokenizer.vocab['<pad>']] * (self.max_seq_len - len(tokens))
            
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        attention_mask = (input_ids != self.tokenizer.vocab['<pad>']).float()
        
        # Mark reasoning tokens for special handling
        reasoning_mask = torch.zeros_like(input_ids)
        for i, token in enumerate(input_ids):
            if token.item() in [
                self.tokenizer.vocab['<think>'], 
                self.tokenizer.vocab['<plan>'],
                self.tokenizer.vocab['<work>']
            ]:
                reasoning_mask[i] = 1
                
        return {
            'input_ids': input_ids,
            'labels': labels, 
            'attention_mask': attention_mask,
            'reasoning_mask': reasoning_mask,
            'has_reasoning': torch.tensor(1.0)
        }


def create_dataloaders(data_dir: str, tokenizer_path: str, 
                      batch_size: int = 8, max_seq_len: int = 512) -> Tuple[DataLoader, DataLoader]:
    """Create train/val dataloaders"""
    
    # Load tokenizer
    tokenizer = LatentReasoningTokenizer()
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
    else:
        raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
    
    # Load text data
    from tokenizer import load_text_data
    all_texts = load_text_data(data_dir)
    
    # Train/val split
    random.shuffle(all_texts)
    split_idx = int(0.9 * len(all_texts))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    print(f"📊 Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Create datasets
    train_dataset = ReasoningDataset(train_texts, tokenizer, max_seq_len)
    val_dataset = ReasoningDataset(val_texts, tokenizer, max_seq_len, reasoning_prob=0.5)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader


def reasoning_collate_fn(batch):
    """Custom collate function for reasoning batches"""
    # Standard collation
    input_ids = torch.stack([item['input_ids'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    result = {
        'input_ids': input_ids,
        'labels': labels,
        'attention_mask': attention_mask
    }
    
    # Add reasoning-specific fields if present
    if 'reasoning_mask' in batch[0]:
        result['reasoning_mask'] = torch.stack([item['reasoning_mask'] for item in batch])
        result['has_reasoning'] = torch.stack([item['has_reasoning'] for item in batch])
        
    return result


if __name__ == "__main__":
    # Test dataset creation
    data_dir = "/home/hp/Documents/HRM/HRM-claude/HRM/data/raw_text"
    tokenizer_path = "/home/hp/Documents/HRM/enhanced_hrm/tokenizer.json"
    
    try:
        train_loader, val_loader = create_dataloaders(data_dir, tokenizer_path, batch_size=4)
        
        print("🎯 Testing data loading...")
        for batch in train_loader:
            print(f"Batch shape: {batch['input_ids'].shape}")
            print(f"Attention mask shape: {batch['attention_mask'].shape}")
            print(f"Sample tokens: {batch['input_ids'][0][:20]}")
            break
            
        print("✅ Dataset pipeline working!")
        
    except FileNotFoundError:
        print("❌ Tokenizer not found. Run tokenizer.py first!")