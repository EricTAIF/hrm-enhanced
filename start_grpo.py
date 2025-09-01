"""
Simple GRPO training starter for Enhanced HRM
- Handles gibberish outputs initially
- Progressive format learning
- Extensive W&B logging
"""

import os
import torch
import wandb
import numpy as np
from datasets import Dataset
from transformers import set_seed
from trl import GRPOTrainer, GRPOConfig

from hrm_hf_wrapper import create_hrm_model_and_tokenizer
from reward_functions import FormatReward, SafeRewardFunction, RewardCombiner, BrevityReward


def create_simple_training_data(num_samples: int = 100) -> Dataset:
    """Create simple format training data"""
    
    examples = []
    
    # Very simple math problems
    for i in range(num_samples // 4):
        a, b = np.random.randint(1, 20, 2)
        examples.append({
            "prompt": f"What is {a} + {b}?",
            "answer": str(a + b),
            "task_type": "math"
        })
    
    # Simple questions
    simple_qa = [
        ("What color is the sky?", "blue"),
        ("How many legs does a dog have?", "four"),
        ("What is the capital of France?", "Paris"),
        ("What comes after 5?", "6")
    ] * (num_samples // 8)
    
    for q, a in simple_qa:
        examples.append({
            "prompt": q,
            "answer": a,
            "task_type": "qa"
        })
        
    # Fill remaining with basic instructions
    remaining = num_samples - len(examples)
    for i in range(remaining):
        examples.append({
            "prompt": "Say hello",
            "answer": "hello",
            "task_type": "instruction"
        })
    
    return Dataset.from_list(examples)


def simple_reward_function(prompts, responses, **kwargs):
    """Simple reward that handles gibberish gracefully"""
    
    rewards = []
    stats = {"format_rewards": [], "length_penalties": []}
    
    for prompt, response in zip(prompts, responses):
        reward = 0.0
        
        # Basic response check
        if not response or len(response.strip()) < 2:
            reward = -2.0  # Heavy penalty for empty
        elif len(response) > 500:
            reward = -1.0  # Penalty for too long (likely repetitive)
        else:
            # Basic format rewards
            if "<think>" in response:
                reward += 1.0
            if "</think>" in response:
                reward += 1.0
            if "<SOLUTION>" in response:
                reward += 2.0
            if "</SOLUTION>" in response:
                reward += 2.0
                
            # Length reward - encourage reasonable lengths
            word_count = len(response.split())
            if 5 <= word_count <= 50:
                reward += 1.0
            elif word_count > 100:
                reward -= 0.5
                
            # Basic coherence - penalize too much repetition
            words = response.split()
            if len(set(words)) / max(len(words), 1) < 0.3:  # Too repetitive
                reward -= 1.0
                
        rewards.append(reward)
        stats["format_rewards"].append(reward)
        stats["length_penalties"].append(len(response.split()))
    
    # Log statistics
    wandb.log({
        "reward/mean": np.mean(rewards),
        "reward/std": np.std(rewards),
        "reward/min": np.min(rewards),
        "reward/max": np.max(rewards),
        "response/avg_length": np.mean(stats["length_penalties"]),
        "response/empty_rate": np.mean([len(r.strip()) < 5 for r in responses])
    })
    
    return rewards


def main():
    # Set seed
    set_seed(42)
    
    # Initialize W&B
    wandb.init(
        project="enhanced-hrm-grpo-simple",
        name="hrm-format-learning",
        config={
            "model_type": "enhanced_hrm",
            "stage": "format_learning",
            "batch_size": 4,
            "num_generations": 8,
            "learning_rate": 5e-5
        }
    )
    
    print("🚀 Starting simple GRPO training for Enhanced HRM")
    
    # Create model and tokenizer
    print("📦 Loading model and tokenizer...")
    model, tokenizer = create_hrm_model_and_tokenizer(
        d_model=256,
        n_heads=4,
        max_cycles=4,  # Start with fewer cycles
        load_checkpoint="checkpoints/best_model.pt" if os.path.exists("checkpoints/best_model.pt") else None
    )
    
    # Create reference model
    print("📋 Creating reference model...")
    ref_model, _ = create_hrm_model_and_tokenizer(
        d_model=256, 
        n_heads=4,
        max_cycles=4
    )
    ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    
    print(f"🧠 Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Create training dataset
    print("📚 Creating training dataset...")
    train_dataset = create_simple_training_data(200)
    print(f"📊 Dataset size: {len(train_dataset)}")
    
    # Show examples
    for i in range(3):
        example = train_dataset[i]
        print(f"Example {i+1}: {example['prompt']} -> {example['answer']}")
    
    # Test current generation quality
    print("\n🎯 Testing current generation:")
    test_prompts = ["What is 2 + 3?", "Say hello", "What color is grass?"]
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 30,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        response = generated_text[len(prompt):].strip()
        print(f"  '{prompt}' -> '{response[:80]}...'")
    
    # GRPO Configuration
    grpo_config = GRPOConfig(
        output_dir="./grpo_simple_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        logging_steps=5,
        save_steps=50,
        num_train_epochs=2,
        warmup_steps=20,
        report_to="wandb",
        remove_unused_columns=False,
        
        # GRPO specific
        num_generations=8,
        max_new_tokens=100,
        temperature=1.0,
        kl_coeff=0.05,
    )
    
    # Create trainer
    print("🏃‍♂️ Creating GRPO trainer...")
    trainer = GRPOTrainer(
        model=model,
        ref_model=ref_model,
        args=grpo_config,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        reward_function=simple_reward_function,
    )
    
    # Train
    print("🚀 Starting GRPO training!")
    try:
        trainer.train()
        print("✅ Training completed!")
        
        # Save model
        model.save_pretrained("./grpo_simple_checkpoints/final")
        tokenizer.save_pretrained("./grpo_simple_checkpoints/final")
        
        # Test after training
        print("\n🎯 Testing after training:")
        for prompt in test_prompts:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
            response = generated_text[len(prompt):].strip()
            print(f"  '{prompt}' -> '{response[:100]}...'")
            
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        wandb.finish()


if __name__ == "__main__":
    main()