"""
Training script for Enhanced HRM with Latent Space Reasoning
- Multi-cycle reasoning with dynamic gating
- Cross-cycle consistency losses  
- Thought-channel supervision
- Entropy-based adaptive training
"""

import os
import math
import json
import argparse
from typing import Dict, Optional
import wandb
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import EnhancedHRM
from dataset import create_dataloaders
from tokenizer import LatentReasoningTokenizer, load_text_data


class ReasoningLoss(nn.Module):
    """Enhanced loss function for reasoning supervision"""
    
    def __init__(self, vocab_size: int, consistency_weight: float = 0.1, 
                 entropy_weight: float = 0.05, thought_weight: float = 0.2):
        super().__init__()
        self.vocab_size = vocab_size
        self.consistency_weight = consistency_weight
        self.entropy_weight = entropy_weight  
        self.thought_weight = thought_weight
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
        
    def forward(self, outputs: Dict, labels: torch.Tensor, 
                attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        logits = outputs['logits']
        B, T, V = logits.shape
        
        # Main next-token prediction loss
        main_loss = self.ce_loss(
            logits.view(-1, V),
            labels.view(-1)
        )
        
        total_loss = main_loss
        loss_dict = {'main_loss': main_loss.item()}
        
        # Cross-cycle consistency loss
        if 'consistency_loss' in outputs:
            consistency_loss = outputs['consistency_loss']
            total_loss = total_loss + self.consistency_weight * consistency_loss
            loss_dict['consistency_loss'] = consistency_loss.item()
            
        # Entropy regularization (encourage confident predictions when appropriate)
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        
        # Reduce entropy for padding tokens, increase for reasoning
        masked_entropy = entropy * attention_mask
        avg_entropy = masked_entropy.sum() / attention_mask.sum()
        
        # Moderate entropy penalty (not too confident, not too uncertain)
        target_entropy = math.log(self.vocab_size) * 0.3  # 30% of max entropy
        entropy_loss = F.mse_loss(avg_entropy, torch.tensor(target_entropy, device=avg_entropy.device))
        
        total_loss = total_loss + self.entropy_weight * entropy_loss
        loss_dict['entropy_loss'] = entropy_loss.item()
        loss_dict['avg_entropy'] = avg_entropy.item()
        
        # Thought supervision (if reasoning masks available)
        if 'reasoning_mask' in outputs and 'thoughts' in outputs:
            # Encourage consistent thought representations during reasoning
            thoughts = outputs['thoughts']
            if len(thoughts) > 1:
                thought_consistency = 0
                for i in range(len(thoughts) - 1):
                    curr_thought = thoughts[i].thought_logits
                    next_thought = thoughts[i + 1].thought_logits
                    thought_consistency += F.mse_loss(curr_thought, next_thought.detach())
                
                thought_consistency = thought_consistency / (len(thoughts) - 1)
                total_loss = total_loss + self.thought_weight * thought_consistency
                loss_dict['thought_loss'] = thought_consistency.item()
        
        loss_dict['total_loss'] = total_loss.item()
        return total_loss, loss_dict


def train_step(model: EnhancedHRM, batch: Dict, loss_fn: ReasoningLoss, 
               optimizer: torch.optim.Optimizer, device: str) -> Dict[str, float]:
    """Single training step"""
    
    model.train()
    
    # Move batch to device
    input_ids = batch['input_ids'].to(device)
    labels = batch['labels'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    
    # Forward pass with variable cycles
    max_cycles = torch.randint(3, 9, (1,)).item()  # Random cycles 3-8
    outputs = model(input_ids, max_cycles=max_cycles, return_thoughts=True)
    
    # Compute loss
    total_loss, loss_dict = loss_fn(outputs, labels, attention_mask)
    
    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Add cycle info
    loss_dict['cycles_used'] = outputs['cycles_used'].item()
    
    return loss_dict


def evaluate(model: EnhancedHRM, val_loader, loss_fn: ReasoningLoss, 
             device: str, max_batches: int = 50) -> Dict[str, float]:
    """Evaluation loop"""
    
    model.eval()
    total_loss = 0
    total_batches = 0
    all_metrics = {}
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if batch_idx >= max_batches:
                break
                
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            # Forward pass
            outputs = model(input_ids, max_cycles=6, return_thoughts=False)
            
            # Compute loss
            total_loss, loss_dict = loss_fn(outputs, labels, attention_mask)
            
            # Accumulate metrics
            for key, value in loss_dict.items():
                if key not in all_metrics:
                    all_metrics[key] = 0
                all_metrics[key] += value
                
            total_batches += 1
    
    # Average metrics
    avg_metrics = {f"val_{k}": v / total_batches for k, v in all_metrics.items()}
    return avg_metrics


def generate_sample(model: EnhancedHRM, tokenizer: LatentReasoningTokenizer,
                   prompt: str, device: str, max_length: int = 100) -> str:
    """Generate text sample for evaluation"""
    
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    # Generate
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids, max_cycles=6)
            logits = outputs['logits']
            
            # Sample next token (with temperature)
            next_token_logits = logits[0, -1, :] / 0.8
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            
            # Stop at EOS or max length
            if next_token.item() == tokenizer.vocab['<eos>']:
                break
    
    # Decode generated sequence
    generated_tokens = input_ids[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/home/hp/Documents/HRM/HRM-claude/HRM/data/raw_text')
    parser.add_argument('--output_dir', default='/home/hp/Documents/HRM/enhanced_hrm')
    parser.add_argument('--vocab_size', type=int, default=8000)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--n_planner_layers', type=int, default=4)
    parser.add_argument('--n_worker_layers', type=int, default=6)
    parser.add_argument('--max_cycles', type=int, default=8)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--warmup_steps', type=int, default=1000)
    parser.add_argument('--save_steps', type=int, default=2000)
    parser.add_argument('--eval_steps', type=int, default=1000)
    parser.add_argument('--use_wandb', action='store_true')
    
    args = parser.parse_args()
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Training Enhanced HRM on {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load/train tokenizer
    tokenizer_path = os.path.join(args.output_dir, 'tokenizer.json')
    tokenizer = LatentReasoningTokenizer(vocab_size=args.vocab_size)
    
    if os.path.exists(tokenizer_path):
        tokenizer.load(tokenizer_path)
        print("📚 Loaded existing tokenizer")
    else:
        print("🔤 Training new tokenizer...")
        texts = load_text_data(args.data_dir)
        tokenizer.train_from_texts(texts)
        tokenizer.save(tokenizer_path)
        print("💾 Tokenizer saved")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        args.data_dir, tokenizer_path, args.batch_size, args.max_seq_len
    )
    
    # Create model
    model = EnhancedHRM(
        vocab_size=len(tokenizer.vocab),
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_planner_layers=args.n_planner_layers,
        n_worker_layers=args.n_worker_layers,
        max_cycles=args.max_cycles,
        max_seq_len=args.max_seq_len
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 Model created with {total_params / 1e6:.1f}M parameters")
    
    # Loss and optimizer
    loss_fn = ReasoningLoss(vocab_size=len(tokenizer.vocab))
    
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )
    
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
    
    # Weights & Biases
    if args.use_wandb:
        wandb.init(
            project="enhanced-hrm",
            config=vars(args),
            name=f"hrm-{args.d_model}d-{args.max_cycles}c"
        )
        wandb.watch(model, log="gradients", log_freq=100)
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    
    print("🏃‍♂️ Starting training...")
    
    for epoch in range(args.epochs):
        print(f"\n📅 Epoch {epoch + 1}/{args.epochs}")
        
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Training step
            loss_dict = train_step(model, batch, loss_fn, optimizer, device)
            
            epoch_loss += loss_dict['total_loss']
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'cycles': f"{loss_dict['cycles_used']:.1f}",
                'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
            # Logging
            if args.use_wandb:
                wandb.log({
                    'train_loss': loss_dict['total_loss'],
                    'train_main_loss': loss_dict.get('main_loss', 0),
                    'train_consistency_loss': loss_dict.get('consistency_loss', 0),
                    'train_entropy_loss': loss_dict.get('entropy_loss', 0),
                    'cycles_used': loss_dict['cycles_used'],
                    'learning_rate': optimizer.param_groups[0]['lr'],
                    'step': global_step
                })
            
            # Evaluation
            if global_step % args.eval_steps == 0:
                print(f"\n🔍 Evaluating at step {global_step}...")
                val_metrics = evaluate(model, val_loader, loss_fn, device)
                
                print(f"Val Loss: {val_metrics['val_total_loss']:.4f}")
                
                if args.use_wandb:
                    wandb.log(val_metrics, step=global_step)
                
                # Generate sample
                sample_prompts = [
                    "Let me think about this problem:",
                    "To solve this step by step:",
                    "The reasoning process is:"
                ]
                
                for prompt in sample_prompts[:1]:  # Just one sample per eval
                    sample_text = generate_sample(model, tokenizer, prompt, device)
                    print(f"🎯 Sample: {sample_text[:200]}...")
                    
                    if args.use_wandb:
                        wandb.log({
                            'sample_generation': wandb.Html(f"<p><b>Prompt:</b> {prompt}</p><p><b>Generated:</b> {sample_text}</p>"),
                        }, step=global_step)
                
                # Save best model
                if val_metrics['val_total_loss'] < best_val_loss:
                    best_val_loss = val_metrics['val_total_loss']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'step': global_step,
                        'val_loss': best_val_loss
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    print("💾 Saved best model")
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                torch.save({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'step': global_step
                }, os.path.join(args.output_dir, f'checkpoint_{global_step}.pt'))
                print(f"💾 Saved checkpoint at step {global_step}")
            
            scheduler.step()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"📊 Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")
    
    print("🎉 Training completed!")
    
    # Final save
    torch.save({
        'model': model.state_dict(),
        'config': vars(args),
        'tokenizer_vocab': tokenizer.vocab
    }, os.path.join(args.output_dir, 'final_model.pt'))
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()