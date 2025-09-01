"""
Simple but effective tokenizer for HRM training
Uses BPE-like approach with special reasoning tokens
"""

import re
import json
from typing import List, Dict, Set
from collections import Counter, defaultdict


class LatentReasoningTokenizer:
    """Tokenizer with special tokens for latent reasoning"""
    
    def __init__(self, vocab_size: int = 8000):
        self.vocab_size = vocab_size
        
        # Special tokens for reasoning
        self.special_tokens = {
            '<pad>': 0,
            '<unk>': 1, 
            '<bos>': 2,
            '<eos>': 3,
            '<think>': 4,     # Start latent reasoning
            '</think>': 5,    # End latent reasoning  
            '<plan>': 6,      # High-level planning
            '</plan>': 7,     # End planning
            '<work>': 8,      # Low-level execution
            '</work>': 9,     # End execution
        }
        
        self.vocab = {}
        self.inv_vocab = {}
        self.merge_rules = []
        
    def train_from_texts(self, texts: List[str]) -> None:
        """Train tokenizer on text corpus"""
        print(f"🔤 Training tokenizer on {len(texts)} texts...")
        
        # Initialize with special tokens
        self.vocab = self.special_tokens.copy()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        # Basic preprocessing - split into words and characters
        word_freqs = Counter()
        
        for text in texts:
            # Clean and split
            words = re.findall(r'\w+|[^\w\s]', text.lower())
            for word in words:
                word_freqs[' '.join(word) + ' </w>'] += 1
        
        # Initialize character vocabulary
        chars = set()
        for word in word_freqs:
            chars.update(word.split())
        
        # Add single characters to vocab
        for char in sorted(chars):
            if char not in self.vocab and len(self.vocab) < self.vocab_size:
                self.vocab[char] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = char
                
        # BPE merging process
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_pairs(word_freqs)
            if not pairs:
                break
                
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Merge the pair
            new_token = ''.join(best_pair)
            if new_token not in self.vocab:
                self.vocab[new_token] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = new_token
                self.merge_rules.append(best_pair)
            
            # Update word frequencies with merged pair
            word_freqs = self._merge_word_freqs(word_freqs, best_pair)
            
        print(f"✅ Tokenizer trained! Vocab size: {len(self.vocab)}")
        
    def _get_pairs(self, word_freqs: Dict[str, int]) -> Counter:
        """Get all adjacent pairs and their frequencies"""
        pairs = Counter()
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i+1])] += freq
        return pairs
        
    def _merge_word_freqs(self, word_freqs: Dict[str, int], pair: tuple) -> Dict[str, int]:
        """Apply merge rule to word frequencies"""
        new_word_freqs = {}
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        
        for word, freq in word_freqs.items():
            new_word = word.replace(bigram, replacement)
            new_word_freqs[new_word] = freq
            
        return new_word_freqs
        
    def encode(self, text: str, add_reasoning_tokens: bool = False) -> List[int]:
        """Encode text to token IDs"""
        if add_reasoning_tokens:
            # Add reasoning structure for training
            text = f"<think> <plan> {text} </plan> <work> [reasoning] </work> </think>"
            
        # Apply BPE encoding
        words = re.findall(r'\w+|[^\w\s]', text.lower())
        tokens = []
        
        for word in words:
            # Handle special tokens
            if word in self.special_tokens:
                tokens.append(self.special_tokens[word])
                continue
                
            # Apply BPE rules
            word_tokens = list(word) + ['</w>']
            
            # Apply merge rules in order
            for pair in self.merge_rules:
                word_tokens = self._apply_merge(word_tokens, pair)
                
            # Convert to IDs
            for token in word_tokens:
                if token in self.vocab:
                    tokens.append(self.vocab[token])
                else:
                    tokens.append(self.vocab['<unk>'])
                    
        return tokens
        
    def _apply_merge(self, tokens: List[str], pair: tuple) -> List[str]:
        """Apply a single merge rule"""
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i+1] == pair[1]:
                new_tokens.append(''.join(pair))
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens
        
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to text"""
        tokens = []
        for token_id in token_ids:
            if token_id in self.inv_vocab:
                tokens.append(self.inv_vocab[token_id])
            else:
                tokens.append('<unk>')
                
        # Join and clean up
        text = ''.join(tokens)
        text = text.replace('</w>', ' ')
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
        
    def save(self, path: str) -> None:
        """Save tokenizer to file"""
        data = {
            'vocab': self.vocab,
            'merge_rules': self.merge_rules,
            'vocab_size': self.vocab_size
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
    def load(self, path: str) -> None:
        """Load tokenizer from file"""
        with open(path, 'r') as f:
            data = json.load(f)
            
        self.vocab = data['vocab']
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.merge_rules = [tuple(rule) for rule in data['merge_rules']]
        self.vocab_size = data['vocab_size']


def load_text_data(data_dir: str) -> List[str]:
    """Load text data from the HRM data directory"""
    import os
    
    texts = []
    text_files = ['test1.txt', 'test2.txt', 'pg30272.txt', 'TinyStories-valid.txt']
    
    for filename in text_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"📖 Loading {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                # Split into sentences/paragraphs
                sentences = re.split(r'[.!?]+', content)
                texts.extend([s.strip() for s in sentences if len(s.strip()) > 20])
    
    # Also load larger files in chunks
    large_files = ['en.txt', 'SimpleStories.txt']
    for filename in large_files:
        filepath = os.path.join(data_dir, filename)
        if os.path.exists(filepath):
            print(f"📚 Loading chunks from {filename}...")
            with open(filepath, 'r', encoding='utf-8') as f:
                chunk_size = 1000000  # 1MB chunks
                chunk = f.read(chunk_size)
                while chunk:
                    sentences = re.split(r'[.!?]+', chunk)
                    texts.extend([s.strip() for s in sentences if len(s.strip()) > 20])
                    chunk = f.read(chunk_size)
                    if len(texts) > 50000:  # Don't load too much for tokenizer training
                        break
                        
    print(f"📊 Loaded {len(texts)} text samples")
    return texts[:50000]  # Cap for training


if __name__ == "__main__":
    # Test the tokenizer
    data_dir = "/home/hp/Documents/HRM/HRM-claude/HRM/data/raw_text"
    
    # Load training data
    texts = load_text_data(data_dir)
    
    # Train tokenizer
    tokenizer = LatentReasoningTokenizer(vocab_size=8000)
    tokenizer.train_from_texts(texts)
    
    # Test encoding/decoding
    test_text = "The model should learn to reason step by step through complex problems."
    
    print(f"\n🧪 Testing tokenizer:")
    print(f"Original: {test_text}")
    
    encoded = tokenizer.encode(test_text)
    print(f"Encoded: {encoded}")
    
    decoded = tokenizer.decode(encoded)
    print(f"Decoded: {decoded}")
    
    # Test with reasoning tokens
    encoded_reasoning = tokenizer.encode(test_text, add_reasoning_tokens=True)
    decoded_reasoning = tokenizer.decode(encoded_reasoning)
    print(f"With reasoning: {decoded_reasoning}")
    
    # Save tokenizer
    tokenizer.save("/home/hp/Documents/HRM/enhanced_hrm/tokenizer.json")
    print(f"💾 Tokenizer saved!")