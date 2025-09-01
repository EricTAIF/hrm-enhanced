"""
Simplified but Working GRPO Training for Enhanced HRM
- Focuses on getting the training running properly
- Progressive improvement from gibberish to reasoning
- W&B tracking
"""

import os
import torch
import wandb
import numpy as np
from typing import List, Dict
from datasets import Dataset
from transformers import set_seed
from trl import GRPOConfig, GRPOTrainer

from hrm_hf_wrapper import create_hrm_model_and_tokenizer
from reward_functions import FormatReward, MathReward


def create_training_data() -> Dataset:
    """Create simple training data"""
    examples = []
    
    # Simple math problems
    for i in range(50):
        a, b = np.random.randint(1, 20, 2)
        examples.append(f"What is {a} + {b}?")
        
    for i in range(30):
        a, b = np.random.randint(1, 10, 2)
        examples.append(f"Calculate {a} * {b}")
        
    # Simple questions
    simple_questions = [
        "What color is the sky?",
        "How many legs does a dog have?", 
        "What comes after 5?",
        "What is 2 + 2?",
        "Say hello"
    ] * 10
    
    examples.extend(simple_questions)
    
    return Dataset.from_dict({"prompt": examples})


def reward_function(prompts: List[str], responses: List[str], **kwargs) -> List[float]:
    """Simple reward function that handles gibberish"""
    
    rewards = []
    format_reward = FormatReward(weight=1.0)
    math_reward = MathReward(weight=2.0)
    
    for prompt, response in zip(prompts, responses):
        reward = 0.0
        
        # Handle completely broken responses
        if not response or len(response.strip()) < 3:
            reward = -2.0
        elif len(response) > 300:  # Too long/repetitive
            reward = -1.0
        else:
            try:
                # Basic format rewards
                if any(tag in response for tag in ["<think>", "<SOLUTION>", "solution", "answer"]):
                    reward += 1.0
                    
                # Math accuracy (simple check)
                if any(word in prompt.lower() for word in ["what is", "calculate", "+"]):
                    # Try to find a number in response
                    numbers = [int(s) for s in response.split() if s.isdigit()]
                    if numbers:
                        reward += 1.0  # Found a number, that's progress
                        
                # Length reward
                word_count = len(response.split())
                if 3 <= word_count <= 30:
                    reward += 0.5
                    
            except:
                reward = -0.5
                
        rewards.append(reward)
    
    # Log stats
    wandb.log({
        "reward_mean": np.mean(rewards),
        "reward_std": np.std(rewards) if len(rewards) > 1 else 0,
        "avg_response_length": np.mean([len(r.split()) for r in responses]),
        "empty_responses": sum(1 for r in responses if len(r.strip()) < 5),
    })
    
    return rewards


def main():
    set_seed(42)
    
    # W&B
    wandb.init(
        project="enhanced-hrm-grpo-simple",
        name="hrm-gibberish-to-reasoning",
        config={"training_type": "grpo", "model": "enhanced_hrm"}
    )
    
    print("🚀 Enhanced HRM GRPO Training")
    print("Starting from gibberish → reasoning")
    
    # Model
    model, tokenizer = create_hrm_model_and_tokenizer(
        d_model=256, n_heads=4, max_cycles=3
    )
    
    print(f"🧠 Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    
    # Test initial quality
    test_prompt = "What is 3 + 4?"
    inputs = tokenizer(test_prompt, return_tensors="pt")
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 20)
    initial_response = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"🎯 Initial: '{test_prompt}' → '{initial_response[len(test_prompt):]}'")
    
    # Dataset
    dataset = create_training_data()
    print(f"📊 Dataset: {len(dataset)} examples")
    
    # GRPO Config
    config = GRPOConfig(
        output_dir="./grpo_simple",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        num_train_epochs=2,
        logging_steps=10,
        save_steps=100,
        warmup_steps=20,
        report_to="wandb"
    )
    
    # Trainer
    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
        reward_function=reward_function,
    )
    
    print("🏃‍♂️ Starting GRPO training...")
    trainer.train()
    
    # Test final quality
    with torch.no_grad():
        generated = model.generate(**inputs, max_length=inputs['input_ids'].shape[1] + 30)
    final_response = tokenizer.decode(generated[0], skip_special_tokens=True)
    print(f"🎯 Final: '{test_prompt}' → '{final_response[len(test_prompt):]}'")
    
    # Save
    model.save_pretrained("./grpo_simple/final")
    tokenizer.save_pretrained("./grpo_simple/final")
    
    print("✅ Training completed!")
    wandb.finish()


if __name__ == "__main__":
    main()