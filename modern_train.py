"""
Modern training script for Enhanced HRM with Phi-3 tokenizer
- Much faster tokenization
- Better text quality
- Latent space reasoning training
"""

import os
import math
import argparse
from typing import Dict, Optional, Tuple, List
from tqdm import tqdm
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import EnhancedHRM
from modern_dataset import create_modern_dataloaders
from modern_tokenizer import ModernReasoningTokenizer


class EnhancedReasoningLoss(nn.Module):
    """Advanced loss function for latent reasoning"""
    
    def __init__(self, vocab_size: int, ignore_index: int = -100):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
        # Loss weights
        self.token_weight = 1.0
        self.consistency_weight = 0.15
        self.entropy_weight = 0.08
        self.confidence_weight = 0.1
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
    def forward(self, outputs: Dict, labels: torch.Tensor, 
                attention_mask: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        
        logits = outputs['logits']
        B, T, V = logits.shape
        
        # Main next-token prediction loss
        token_loss = self.ce_loss(logits.view(-1, V), labels.view(-1))
        total_loss = self.token_weight * token_loss
        
        loss_metrics = {
            'token_loss': token_loss.item(),
            'perplexity': torch.exp(token_loss).item()
        }
        
        # Cross-cycle consistency loss
        if 'consistency_loss' in outputs:
            consistency_loss = outputs['consistency_loss']
            total_loss = total_loss + self.consistency_weight * consistency_loss
            loss_metrics['consistency_loss'] = consistency_loss.item()
        
        # Entropy regularization for better reasoning
        with torch.no_grad():
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            
            # Mask out padding tokens
            valid_positions = (labels != self.ignore_index).float()
            if valid_positions.sum() > 0:
                avg_entropy = (entropy * valid_positions).sum() / valid_positions.sum()
                loss_metrics['entropy'] = avg_entropy.item()
                
                # Adaptive entropy penalty
                target_entropy = math.log(V) * 0.25  # Target 25% of max entropy
                entropy_penalty = F.mse_loss(avg_entropy, torch.tensor(target_entropy, device=avg_entropy.device))
                total_loss = total_loss + self.entropy_weight * entropy_penalty
                loss_metrics['entropy_penalty'] = entropy_penalty.item()
        
        # Confidence regularization (from thought channel)
        if 'thoughts' in outputs and len(outputs['thoughts']) > 0:
            # Average confidence across cycles
            confidences = [thought.confidence.mean() for thought in outputs['thoughts']]
            if confidences:
                avg_confidence = torch.stack(confidences).mean()
                
                # Encourage reasonable confidence (not too low, not overconfident)
                target_confidence = 0.7
                confidence_loss = F.mse_loss(avg_confidence, torch.tensor(target_confidence, device=avg_confidence.device))
                total_loss = total_loss + self.confidence_weight * confidence_loss
                loss_metrics['confidence_loss'] = confidence_loss.item()
                loss_metrics['avg_confidence'] = avg_confidence.item()
        
        loss_metrics['total_loss'] = total_loss.item()
        loss_metrics['cycles_used'] = outputs.get('cycles_used', torch.tensor(1)).item()
        
        return total_loss, loss_metrics


def train_step(model: EnhancedHRM, batch: Dict, loss_fn: EnhancedReasoningLoss, 
               optimizer: torch.optim.Optimizer, device: str, max_cycles: int = 6) -> Dict[str, float]:
    """Single training step with gradient accumulation"""
    
    model.train()
    
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Variable number of reasoning cycles (3-8)
    num_cycles = torch.randint(3, max_cycles + 1, (1,)).item()
    
    # Forward pass
    outputs = model(input_ids, max_cycles=num_cycles, return_thoughts=True)
    
    # Compute loss
    total_loss, metrics = loss_fn(outputs, labels, attention_mask)
    
    # Backward pass
    total_loss.backward()
    
    # Gradient clipping for stability
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    metrics['grad_norm'] = grad_norm.item()
    
    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()
    
    return metrics


@torch.no_grad()
def evaluate_model(model: EnhancedHRM, val_loader, loss_fn: EnhancedReasoningLoss, 
                  device: str, max_batches: int = 30) -> Dict[str, float]:
    """Evaluation with consistent reasoning cycles"""
    
    model.eval()
    
    all_metrics = {}
    total_samples = 0
    
    for batch_idx, batch in enumerate(val_loader):
        if batch_idx >= max_batches:
            break
            
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Use fixed cycles for consistent evaluation
        outputs = model(input_ids, max_cycles=6, return_thoughts=False)
        
        # Compute metrics
        _, metrics = loss_fn(outputs, labels, attention_mask)
        
        batch_size = input_ids.size(0)
        
        # Accumulate metrics
        for key, value in metrics.items():
            if key not in all_metrics:
                all_metrics[key] = 0
            all_metrics[key] += value * batch_size
        
        total_samples += batch_size
    
    # Average metrics
    avg_metrics = {f"val_{k}": v / total_samples for k, v in all_metrics.items()}
    return avg_metrics


def generate_samples(model: EnhancedHRM, tokenizer: ModernReasoningTokenizer,
                    device: str, max_length: int = 80) -> List[str]:
    """Generate text samples for evaluation"""
    
    model.eval()
    
    prompts = [
        "Let me think about this problem:",
        "To solve this step by step:",
        "The reasoning process involves:",
        "<think> <plan>",
        "Once upon a time"
    ]
    
    samples = []
    
    with torch.no_grad():
        for prompt in prompts:
            # Encode prompt
            input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
            
            # Generate
            for _ in range(max_length):
                outputs = model(input_ids, max_cycles=4)
                logits = outputs['logits']
                
                # Sample next token with temperature
                next_token_logits = logits[0, -1, :] / 0.8
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Append token
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                
                # Stop at EOS or if we hit reasoning tokens
                if next_token.item() == tokenizer.tokenizer.eos_token_id:
                    break
                    
                # Avoid too long generations
                if input_ids.size(1) > len(tokenizer.encode(prompt)) + max_length:
                    break
            
            # Decode sample
            generated_ids = input_ids[0].cpu().tolist()
            sample_text = tokenizer.decode(generated_ids)
            samples.append(f"**Prompt:** {prompt}\n**Generated:** {sample_text}\n")
    
    return samples


def main():
    parser = argparse.ArgumentParser(description="Train Enhanced HRM with Modern Tokenizer")
    parser.add_argument('--data_dir', default='/home/hp/Documents/HRM/HRM-claude/HRM/data/raw_text')
    parser.add_argument('--output_dir', default='/home/hp/Documents/HRM/enhanced_hrm/checkpoints')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--n_planner_layers', type=int, default=4, help='Planner layers')
    parser.add_argument('--n_worker_layers', type=int, default=6, help='Worker layers') 
    parser.add_argument('--max_cycles', type=int, default=8, help='Max reasoning cycles')
    parser.add_argument('--max_seq_len', type=int, default=512, help='Max sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Warmup steps')
    parser.add_argument('--eval_steps', type=int, default=500, help='Evaluation frequency')
    parser.add_argument('--save_steps', type=int, default=1000, help='Save frequency')
    parser.add_argument('--max_samples', type=int, default=10000, help='Max training samples')
    parser.add_argument('--compile', action='store_true', help='Use torch.compile')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Enhanced HRM Training on {device}")
    print(f"🔧 Args: {vars(args)}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create dataloaders with modern tokenizer
    print("📚 Creating dataloaders...")
    train_loader, val_loader, tokenizer = create_modern_dataloaders(
        args.data_dir, 
        batch_size=args.batch_size, 
        max_seq_len=args.max_seq_len,
        max_samples=args.max_samples
    )
    
    vocab_size = tokenizer.vocab_size
    print(f"📖 Vocabulary size: {vocab_size}")
    
    # Create enhanced model
    print("🧠 Creating Enhanced HRM...")
    model = EnhancedHRM(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_planner_layers=args.n_planner_layers,
        n_worker_layers=args.n_worker_layers,
        max_cycles=args.max_cycles,
        max_seq_len=args.max_seq_len,
        dropout=0.1
    ).to(device)
    
    # Compile for speed (if requested and supported)
    if args.compile and hasattr(torch, 'compile'):
        print("⚡ Compiling model...")
        model = torch.compile(model)
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🎯 Model parameters: {total_params / 1e6:.1f}M")
    
    # Loss function and optimizer
    loss_fn = EnhancedReasoningLoss(vocab_size=vocab_size, ignore_index=-100)
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8
    )
    
    # Learning rate scheduler
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.learning_rate * 0.1)
    
    # Training loop
    print(f"🏃‍♂️ Starting training for {args.epochs} epochs ({total_steps} steps)")
    
    global_step = 0
    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        
        # Training phase
        model.train()
        epoch_metrics = {}
        
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            metrics = train_step(model, batch, loss_fn, optimizer, device, args.max_cycles)
            
            # Accumulate metrics
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                epoch_metrics[key].append(value)
            
            global_step += 1
            
            # Update progress
            current_loss = metrics['total_loss']
            current_ppl = metrics.get('perplexity', 0)
            cycles = metrics.get('cycles_used', 0)
            
            progress_bar.set_postfix({
                'loss': f"{current_loss:.4f}",
                'ppl': f"{current_ppl:.2f}",
                'cycles': f"{cycles:.1f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                print(f"\n🔍 Evaluating at step {global_step}...")
                val_metrics = evaluate_model(model, val_loader, loss_fn, device)
                
                val_loss = val_metrics['val_total_loss']
                val_ppl = val_metrics.get('val_perplexity', 0)
                
                print(f"📊 Val Loss: {val_loss:.4f}, Val PPL: {val_ppl:.2f}")
                
                # Generate samples
                samples = generate_samples(model, tokenizer, device)
                print("\n🎯 Generated samples:")
                for i, sample in enumerate(samples[:2]):
                    print(f"{sample}")
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    checkpoint = {
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'step': global_step,
                        'val_loss': val_loss,
                        'args': vars(args)
                    }
                    torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
                    print(f"💾 Saved best model (val_loss: {val_loss:.4f})")
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                checkpoint = {
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': global_step,
                    'args': vars(args)
                }
                torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_step_{global_step}.pt'))
                print(f"\n💾 Saved checkpoint at step {global_step}")
            
            scheduler.step()
        
        # Epoch summary
        avg_epoch_metrics = {k: sum(v) / len(v) for k, v in epoch_metrics.items()}
        print(f"\n📈 Epoch {epoch+1} Summary:")
        print(f"  Loss: {avg_epoch_metrics['total_loss']:.4f}")
        print(f"  Perplexity: {avg_epoch_metrics.get('perplexity', 0):.2f}")
        print(f"  Avg Cycles: {avg_epoch_metrics.get('cycles_used', 0):.1f}")
        print(f"  Grad Norm: {avg_epoch_metrics.get('grad_norm', 0):.3f}")
    
    # Final evaluation and save
    print("\n🏁 Final evaluation...")
    final_val_metrics = evaluate_model(model, val_loader, loss_fn, device, max_batches=100)
    
    print("📊 Final validation metrics:")
    for key, value in final_val_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Final samples
    print("\n🎯 Final generated samples:")
    final_samples = generate_samples(model, tokenizer, device, max_length=100)
    for sample in final_samples:
        print(sample)
    
    # Save final model
    final_checkpoint = {
        'model_state_dict': model.state_dict(),
        'final_metrics': final_val_metrics,
        'args': vars(args),
        'vocab_size': vocab_size,
        'tokenizer_name': 'microsoft/Phi-3-mini-4k-instruct'
    }
    torch.save(final_checkpoint, os.path.join(args.output_dir, 'final_enhanced_hrm.pt'))
    
    # Save config for easy loading
    config = {
        'vocab_size': vocab_size,
        'd_model': args.d_model,
        'n_heads': args.n_heads,
        'n_planner_layers': args.n_planner_layers,
        'n_worker_layers': args.n_worker_layers,
        'max_cycles': args.max_cycles,
        'max_seq_len': args.max_seq_len,
        'tokenizer_name': 'microsoft/Phi-3-mini-4k-instruct'
    }
    
    with open(os.path.join(args.output_dir, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print("🎉 Training completed successfully!")
    print(f"💾 Models saved in: {args.output_dir}")
    print("✨ LATENT SPACE REASONING MODEL READY! ✨")


if __name__ == "__main__":
    main()