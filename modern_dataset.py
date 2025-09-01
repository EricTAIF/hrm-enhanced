"""
Modern dataset using Phi-3 tokenizer for Enhanced HRM training
"""

import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from modern_tokenizer import ModernReasoningTokenizer


class ModernReasoningDataset(Dataset):
    """Dataset using modern tokenizer for HRM training"""
    
    def __init__(self, texts: List[str], tokenizer: ModernReasoningTokenizer,
                 max_seq_len: int = 512, reasoning_prob: float = 0.3):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.reasoning_prob = reasoning_prob
        
        print(f"🔄 Processing {len(texts)} texts...")
        
        # Filter and process texts
        self.processed_texts = []
        for text in texts:
            if len(text.strip()) > 20:  # Filter very short texts
                self.processed_texts.append(text.strip())
        
        print(f"✅ Dataset ready with {len(self.processed_texts)} samples")
        
    def __len__(self):
        return len(self.processed_texts)
        
    def __getitem__(self, idx):
        text = self.processed_texts[idx]
        
        # Randomly add reasoning structure
        if random.random() < self.reasoning_prob:
            text = f"<think> <plan> {text} </plan> <work> Let me process this step by step. </work> </think>"
        
        # Tokenize
        encoding = self.tokenizer.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create labels (shifted input for next-token prediction)
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100  # Ignore padding tokens in loss
        
        # Shift for causal language modeling
        if len(input_ids) > 1:
            input_ids = input_ids[:-1]
            labels = labels[1:]
            attention_mask = attention_mask[:-1]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask
        }


def load_text_files(data_dir: str, max_samples: int = 20000) -> List[str]:
    """Load text data from directory"""
    texts = []
    
    # Load smaller files completely
    small_files = ['test1.txt', 'test2.txt', 'pg30272.txt']
    for filename in small_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"📖 Loading {filename}...")
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                # Split into sentences
                sentences = content.replace('\n', ' ').split('. ')
                texts.extend([s.strip() + '.' for s in sentences if len(s.strip()) > 20])
    
    # Load larger files in chunks
    large_files = ['TinyStories-valid.txt', 'en.txt', 'SimpleStories.txt']
    for filename in large_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath) and len(texts) < max_samples:
            print(f"📚 Loading chunks from {filename}...")
            try:
                with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                    chunk_size = 100000  # 100KB chunks
                    while len(texts) < max_samples:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        
                        # Split into sentences
                        sentences = chunk.replace('\n', ' ').split('. ')
                        new_texts = [s.strip() + '.' for s in sentences if len(s.strip()) > 20]
                        texts.extend(new_texts)
                        
                        if len(texts) % 1000 == 0:
                            print(f"  Loaded {len(texts)} samples so far...")
            except Exception as e:
                print(f"  Error loading {filename}: {e}")
                continue
                
    print(f"📊 Total loaded: {len(texts)} text samples")
    return texts[:max_samples]


def create_modern_dataloaders(data_dir: str, batch_size: int = 8, 
                             max_seq_len: int = 512, max_samples: int = 20000) -> Tuple[DataLoader, DataLoader]:
    """Create dataloaders with modern tokenizer"""
    
    # Initialize tokenizer
    tokenizer = ModernReasoningTokenizer()
    
    # Load text data
    all_texts = load_text_files(data_dir, max_samples)
    
    # Train/val split
    random.shuffle(all_texts)
    split_idx = int(0.9 * len(all_texts))
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    print(f"📊 Train: {len(train_texts)}, Val: {len(val_texts)}")
    
    # Create datasets
    train_dataset = ModernReasoningDataset(train_texts, tokenizer, max_seq_len, reasoning_prob=0.3)
    val_dataset = ModernReasoningDataset(val_texts, tokenizer, max_seq_len, reasoning_prob=0.5)
    
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
        num_workers=1,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, tokenizer


if __name__ == "__main__":
    # Test modern dataset
    data_dir = "/home/hp/Documents/HRM/HRM-claude/HRM/data/raw_text"
    
    print("🚀 Testing modern dataset pipeline...")
    
    try:
        train_loader, val_loader, tokenizer = create_modern_dataloaders(
            data_dir, batch_size=4, max_seq_len=256, max_samples=1000
        )
        
        print("🎯 Testing data loading...")
        for batch_idx, batch in enumerate(train_loader):
            print(f"Batch {batch_idx}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            
            # Decode first sample
            sample_ids = batch['input_ids'][0]
            sample_text = tokenizer.decode(sample_ids[sample_ids != tokenizer.pad_token_id].tolist())
            print(f"  Sample text: {sample_text[:100]}...")
            
            if batch_idx >= 2:
                break
                
        print("✅ Modern dataset pipeline working perfectly!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()