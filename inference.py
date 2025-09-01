"""
Interactive inference script for Enhanced HRM with Latent Space Reasoning
- Real-time text generation with multi-cycle reasoning
- Visualize reasoning process and cycle usage
- Interactive chat interface
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
import argparse

from model import EnhancedHRM
from modern_tokenizer import ModernReasoningTokenizer


class HRMInference:
    """Interactive inference wrapper for Enhanced HRM"""
    
    def __init__(self, checkpoint_path: str, config_path: Optional[str] = None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"🚀 Loading Enhanced HRM on {self.device}")
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Default config (adjust if needed)
            self.config = {
                'vocab_size': 32019,
                'd_model': 256,
                'n_heads': 4,
                'n_planner_layers': 2,
                'n_worker_layers': 3,
                'max_cycles': 6,
                'max_seq_len': 512
            }
        
        # Load tokenizer
        self.tokenizer = ModernReasoningTokenizer()
        print(f"📚 Tokenizer loaded (vocab size: {self.tokenizer.vocab_size})")
        
        # Create and load model (filter out non-model config)
        model_config = {k: v for k, v in self.config.items() 
                       if k in ['vocab_size', 'd_model', 'n_heads', 'n_planner_layers', 
                               'n_worker_layers', 'max_cycles', 'max_seq_len', 'thought_dim', 'dropout']}
        self.model = EnhancedHRM(**model_config).to(self.device)
        
        # Load checkpoint
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Model loaded from {checkpoint_path}")
                if 'val_loss' in checkpoint:
                    print(f"📊 Checkpoint validation loss: {checkpoint['val_loss']:.4f}")
            else:
                # Try direct state dict
                self.model.load_state_dict(checkpoint)
                print(f"✅ Model state loaded from {checkpoint_path}")
        else:
            print(f"❌ Checkpoint not found: {checkpoint_path}")
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.model.eval()
        
        # Model info
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"🧠 Model parameters: {total_params / 1e6:.1f}M")
        print(f"🔧 Max reasoning cycles: {self.config['max_cycles']}")
        
    def generate(self, prompt: str, max_length: int = 100, temperature: float = 0.8,
                top_k: int = 50, max_cycles: Optional[int] = None, show_reasoning: bool = True) -> Dict:
        """Generate text with reasoning visualization"""
        
        max_cycles = max_cycles or self.config['max_cycles']
        
        print(f"\n🎯 Generating with prompt: '{prompt}'")
        print(f"⚙️  Settings: max_length={max_length}, temp={temperature}, cycles={max_cycles}")
        
        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        original_length = input_ids.size(1)
        
        generated_tokens = []
        reasoning_info = []
        
        with torch.no_grad():
            for step in range(max_length):
                # Forward pass with reasoning
                outputs = self.model(input_ids, max_cycles=max_cycles, return_thoughts=True)
                
                logits = outputs['logits'][0, -1, :]  # Last token logits
                cycles_used = outputs.get('cycles_used', 1)
                
                # Temperature scaling
                if temperature > 0:
                    logits = logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(logits, top_k)
                    logits_filtered = torch.full_like(logits, float('-inf'))
                    logits_filtered.scatter_(0, top_k_indices, top_k_logits)
                    logits = logits_filtered
                
                # Sample next token
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                next_token_id = next_token.item()
                
                # Track reasoning info
                reasoning_info.append({
                    'step': step,
                    'cycles_used': cycles_used.item() if torch.is_tensor(cycles_used) else cycles_used,
                    'token_id': next_token_id,
                    'token': self.tokenizer.decode([next_token_id]),
                    'confidence': probs[next_token_id].item(),
                    'entropy': -torch.sum(probs * torch.log(probs + 1e-10)).item()
                })
                
                if show_reasoning and step % 10 == 0:
                    print(f"  Step {step}: cycles={cycles_used}, token='{self.tokenizer.decode([next_token_id])}', conf={probs[next_token_id]:.3f}")
                
                # Add token to sequence
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                generated_tokens.append(next_token_id)
                
                # Stop conditions
                if next_token_id == self.tokenizer.tokenizer.eos_token_id:
                    print("🛑 EOS token reached")
                    break
                    
                if input_ids.size(1) > self.config['max_seq_len'] - 10:
                    print("📏 Max sequence length reached")
                    break
        
        # Decode full sequence
        full_text = self.tokenizer.decode(input_ids[0].cpu().tolist())
        generated_text = self.tokenizer.decode(generated_tokens)
        
        # Calculate statistics
        avg_cycles = sum(info['cycles_used'] for info in reasoning_info) / len(reasoning_info)
        avg_confidence = sum(info['confidence'] for info in reasoning_info) / len(reasoning_info)
        avg_entropy = sum(info['entropy'] for info in reasoning_info) / len(reasoning_info)
        
        return {
            'prompt': prompt,
            'generated_text': generated_text,
            'full_text': full_text,
            'tokens_generated': len(generated_tokens),
            'reasoning_info': reasoning_info,
            'statistics': {
                'avg_cycles': avg_cycles,
                'avg_confidence': avg_confidence,
                'avg_entropy': avg_entropy,
                'total_steps': len(reasoning_info)
            }
        }
    
    def interactive_chat(self):
        """Interactive chat mode"""
        print("\n" + "="*60)
        print("🧠 Enhanced HRM - Interactive Reasoning Chat")
        print("="*60)
        print("Commands:")
        print("  /help - Show help")
        print("  /settings - Change generation settings")
        print("  /reasoning - Toggle reasoning visualization")
        print("  /quit - Exit")
        print("="*60)
        
        # Default settings
        settings = {
            'max_length': 80,
            'temperature': 0.8,
            'top_k': 50,
            'max_cycles': 6,
            'show_reasoning': True
        }
        
        while True:
            try:
                prompt = input("\n👤 You: ").strip()
                
                if not prompt:
                    continue
                    
                if prompt == "/quit":
                    print("👋 Goodbye!")
                    break
                    
                elif prompt == "/help":
                    print("📖 Help:")
                    print("  - Just type any text to generate a continuation")
                    print("  - The model will use multi-cycle reasoning")
                    print("  - Use /settings to adjust generation parameters")
                    continue
                    
                elif prompt == "/settings":
                    print("⚙️ Current settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    
                    print("\nChange setting (press Enter to skip):")
                    for key in settings:
                        new_val = input(f"  {key} [{settings[key]}]: ").strip()
                        if new_val:
                            try:
                                if key == 'show_reasoning':
                                    settings[key] = new_val.lower() in ['true', 'yes', '1']
                                else:
                                    settings[key] = type(settings[key])(new_val)
                            except ValueError:
                                print(f"    Invalid value for {key}")
                    continue
                    
                elif prompt == "/reasoning":
                    settings['show_reasoning'] = not settings['show_reasoning']
                    print(f"🔧 Reasoning visualization: {'ON' if settings['show_reasoning'] else 'OFF'}")
                    continue
                
                # Generate response
                result = self.generate(prompt, **settings)
                
                print(f"\n🤖 HRM: {result['generated_text']}")
                
                # Show statistics
                stats = result['statistics']
                print(f"📊 Stats: {stats['total_steps']} tokens, {stats['avg_cycles']:.1f} avg cycles, {stats['avg_confidence']:.3f} confidence")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def benchmark_reasoning(self, prompts: List[str]) -> Dict:
        """Benchmark reasoning on multiple prompts"""
        print(f"\n🏃‍♂️ Benchmarking reasoning on {len(prompts)} prompts...")
        
        results = []
        total_cycles = 0
        total_tokens = 0
        
        for i, prompt in enumerate(prompts):
            print(f"\n[{i+1}/{len(prompts)}] {prompt[:50]}...")
            
            result = self.generate(prompt, max_length=50, show_reasoning=False)
            results.append(result)
            
            stats = result['statistics']
            total_cycles += stats['avg_cycles'] * stats['total_steps']
            total_tokens += stats['total_steps']
            
            print(f"  Generated: {result['generated_text'][:80]}...")
            print(f"  Cycles: {stats['avg_cycles']:.1f}, Confidence: {stats['avg_confidence']:.3f}")
        
        avg_cycles_per_token = total_cycles / total_tokens if total_tokens > 0 else 0
        
        print(f"\n📈 Benchmark Results:")
        print(f"  Total prompts: {len(prompts)}")
        print(f"  Total tokens generated: {total_tokens}")
        print(f"  Average cycles per token: {avg_cycles_per_token:.2f}")
        
        return {
            'results': results,
            'total_tokens': total_tokens,
            'avg_cycles_per_token': avg_cycles_per_token
        }


def main():
    parser = argparse.ArgumentParser(description="Enhanced HRM Inference")
    parser.add_argument('--checkpoint', default='checkpoints/best_model.pt', help='Model checkpoint path')
    parser.add_argument('--config', default='checkpoints/model_config.json', help='Model config path')
    parser.add_argument('--mode', choices=['chat', 'generate', 'benchmark'], default='chat', help='Inference mode')
    parser.add_argument('--prompt', type=str, help='Prompt for generation mode')
    parser.add_argument('--max_length', type=int, default=100, help='Max generation length')
    parser.add_argument('--temperature', type=float, default=0.8, help='Generation temperature')
    parser.add_argument('--max_cycles', type=int, default=6, help='Max reasoning cycles')
    
    args = parser.parse_args()
    
    # Initialize inference
    try:
        hrm = HRMInference(args.checkpoint, args.config)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    if args.mode == 'chat':
        hrm.interactive_chat()
        
    elif args.mode == 'generate':
        if not args.prompt:
            print("❌ Please provide --prompt for generation mode")
            return
            
        result = hrm.generate(
            args.prompt, 
            max_length=args.max_length,
            temperature=args.temperature,
            max_cycles=args.max_cycles
        )
        
        print(f"\n🎯 Generated text:")
        print(f"'{result['generated_text']}'")
        
        stats = result['statistics']
        print(f"\n📊 Statistics:")
        print(f"  Tokens generated: {stats['total_steps']}")
        print(f"  Average cycles: {stats['avg_cycles']:.2f}")
        print(f"  Average confidence: {stats['avg_confidence']:.3f}")
        print(f"  Average entropy: {stats['avg_entropy']:.3f}")
        
    elif args.mode == 'benchmark':
        test_prompts = [
            "Let me think about this problem:",
            "To solve this step by step:",
            "The reasoning process involves:",
            "Once upon a time",
            "In order to understand",
            "The key insight is",
            "Step by step analysis:",
            "<think> <plan>",
            "Mathematical reasoning:",
            "The solution requires"
        ]
        
        hrm.benchmark_reasoning(test_prompts)


if __name__ == "__main__":
    main()