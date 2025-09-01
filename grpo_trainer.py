"""
GRPO Training Pipeline for Enhanced HRM
- Handles models that start with gibberish
- Progressive curriculum from format learning to reasoning
- Extensive W&B logging and metrics
- Robust reward functions for unstructured outputs
"""

import os
import re
import json
import torch
import wandb
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from transformers import TrainingArguments, set_seed
from trl import GRPOTrainer, GRPOConfig
from datasets import Dataset, load_dataset, concatenate_datasets

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    UNSLOTH_AVAILABLE = False

from hrm_hf_wrapper import create_hrm_model_and_tokenizer, EnhancedHRMForCausalLM
from reward_functions import RewardCombiner, create_math_reward_combiner, create_code_reward_combiner
from reward_functions import FormatReward, MathReward, CodeReward, LatentReasoningReward, BrevityReward


@dataclass
class HRMGRPOConfig:
    """Configuration for HRM GRPO training"""
    
    # Model config
    model_name: str = "enhanced_hrm"
    d_model: int = 256
    n_heads: int = 4
    n_planner_layers: int = 2
    n_worker_layers: int = 3
    max_cycles: int = 6
    max_seq_len: int = 512
    
    # Training config
    num_generations: int = 8  # GRPO group size
    max_new_tokens: int = 200
    temperature: float = 1.0
    
    # GRPO specific
    loss_type: str = "grpo"  # or "dr_grpo", "dapo"
    kl_coeff: float = 0.05
    learning_rate: float = 5e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 2
    num_train_epochs: int = 3
    
    # Curriculum stages
    warmup_steps: int = 200  # SFT format warmup
    format_focus_steps: int = 500  # Focus on format learning
    reasoning_steps: int = 1000  # Full reasoning training
    
    # Data mixing
    math_ratio: float = 0.6
    code_ratio: float = 0.3
    general_ratio: float = 0.1
    
    # Reward weights (will adjust during curriculum)
    format_weight: float = 2.0  # Start high for gibberish models
    correctness_weight: float = 1.0  # Start low, increase later
    latent_weight: float = 0.1
    brevity_weight: float = 0.1


class SafeRewardFunction:
    """Wrapper that handles malformed/gibberish responses safely"""
    
    def __init__(self, reward_fn):
        self.reward_fn = reward_fn
        self.name = getattr(reward_fn, 'name', 'unknown')
        
    def __call__(self, response: str, reference: Optional[str] = None, **kwargs) -> float:
        try:
            # Handle completely empty or very short responses
            if not response or len(response.strip()) < 3:
                return -1.0
                
            # Basic structure check
            if len(response) > 2000:  # Too long, likely repetitive gibberish
                return -0.5
                
            return self.reward_fn(response, reference, **kwargs)
            
        except Exception as e:
            # If reward function crashes on gibberish, return small negative
            print(f"⚠️  Reward function {self.name} failed: {e}")
            return -0.1


def create_format_examples() -> List[Dict[str, str]]:
    """Create simple format examples for initial SFT warmup"""
    return [
        {
            "instruction": "What is 2 + 2?",
            "output": "<think>I need to add 2 and 2.</think>\n<SOLUTION>4</SOLUTION>"
        },
        {
            "instruction": "What color is the sky?", 
            "output": "<think>The sky appears blue during the day.</think>\n<SOLUTION>Blue</SOLUTION>"
        },
        {
            "instruction": "Write a function to add two numbers:",
            "output": "<think>I need to write a simple addition function.</think>\n<SOLUTION>\ndef add(a, b):\n    return a + b\n</SOLUTION>"
        },
        {
            "instruction": "Solve: 5 * 6",
            "output": "<think>5 times 6 equals 30.</think>\n<SOLUTION>30</SOLUTION>"
        },
        {
            "instruction": "What is the capital of France?",
            "output": "<think>The capital of France is Paris.</think>\n<SOLUTION>Paris</SOLUTION>"
        }
    ] * 10  # Repeat for more examples


def load_math_dataset(num_samples: int = 1000) -> Dataset:
    """Load GSM8K dataset for math reasoning"""
    try:
        dataset = load_dataset("openai/gsm8k", "main", split="train")
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        def format_math_example(example):
            return {
                "prompt": f"Solve this math problem step by step:\n{example['question']}",
                "reference_answer": str(example['answer']).split('\n')[-1].split('#### ')[-1],
                "task_type": "math"
            }
            
        return dataset.map(format_math_example)
        
    except Exception as e:
        print(f"⚠️  Could not load GSM8K: {e}")
        # Fallback to simple math problems
        simple_math = [
            {"prompt": "What is 15 + 27?", "reference_answer": "42", "task_type": "math"},
            {"prompt": "Calculate 8 * 9", "reference_answer": "72", "task_type": "math"},
            {"prompt": "What is 100 - 37?", "reference_answer": "63", "task_type": "math"},
            {"prompt": "Solve: 144 / 12", "reference_answer": "12", "task_type": "math"},
        ] * (num_samples // 4)
        
        return Dataset.from_list(simple_math)


def load_code_dataset(num_samples: int = 500) -> Dataset:
    """Load HumanEval dataset for code reasoning"""
    try:
        dataset = load_dataset("openai/humaneval", split="test")
        dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        def format_code_example(example):
            return {
                "prompt": f"Complete this Python function:\n{example['prompt']}",
                "reference_answer": example['canonical_solution'],
                "task_type": "code",
                "test_cases": example.get('test', [])
            }
            
        return dataset.map(format_code_example)
        
    except Exception as e:
        print(f"⚠️  Could not load HumanEval: {e}")
        # Fallback to simple coding problems
        simple_code = [
            {
                "prompt": "Write a function that returns the square of a number:",
                "reference_answer": "def square(x):\n    return x * x",
                "task_type": "code",
                "test_cases": [{"test": "assert square(4) == 16"}]
            },
            {
                "prompt": "Write a function that checks if a number is even:",
                "reference_answer": "def is_even(n):\n    return n % 2 == 0",
                "task_type": "code", 
                "test_cases": [{"test": "assert is_even(4) == True"}]
            }
        ] * (num_samples // 2)
        
        return Dataset.from_list(simple_code)


def create_curriculum_dataset(config: HRMGRPOConfig, stage: str = "warmup") -> Dataset:
    """Create dataset based on curriculum stage"""
    
    if stage == "warmup":
        # Pure format learning
        examples = create_format_examples()
        return Dataset.from_list([
            {"prompt": ex["instruction"], "reference_answer": "", "task_type": "format"}
            for ex in examples
        ])
        
    elif stage == "format_focus":
        # Mix of format examples and simple problems
        format_examples = create_format_examples()[:50]
        math_simple = load_math_dataset(100)
        
        format_data = Dataset.from_list([
            {"prompt": ex["instruction"], "reference_answer": "", "task_type": "format"}
            for ex in format_examples
        ])
        
        return concatenate_datasets([format_data, math_simple])
        
    else:  # "reasoning"
        # Full curriculum
        math_samples = int(config.math_ratio * 1000)
        code_samples = int(config.code_ratio * 1000) 
        
        datasets = []
        
        if math_samples > 0:
            datasets.append(load_math_dataset(math_samples))
            
        if code_samples > 0:
            datasets.append(load_code_dataset(code_samples))
            
        return concatenate_datasets(datasets) if datasets else load_math_dataset(500)


def create_adaptive_reward_combiner(stage: str, config: HRMGRPOConfig) -> RewardCombiner:
    """Create reward combiner that adapts based on training stage"""
    
    if stage == "warmup":
        # Focus heavily on format for gibberish models
        return RewardCombiner([
            SafeRewardFunction(FormatReward(weight=5.0)),  # Very high format reward
            SafeRewardFunction(BrevityReward(weight=0.5))
        ])
        
    elif stage == "format_focus":
        # Still prioritize format but add some correctness
        return RewardCombiner([
            SafeRewardFunction(FormatReward(weight=3.0)),
            SafeRewardFunction(MathReward(weight=1.0)),
            SafeRewardFunction(LatentReasoningReward(weight=0.2)),
            SafeRewardFunction(BrevityReward(weight=0.3))
        ])
        
    else:  # "reasoning"
        # Full reward system
        return RewardCombiner([
            SafeRewardFunction(FormatReward(weight=config.format_weight)),
            SafeRewardFunction(MathReward(weight=config.correctness_weight * 3.0)),
            SafeRewardFunction(CodeReward(weight=config.correctness_weight * 3.0)),
            SafeRewardFunction(LatentReasoningReward(weight=config.latent_weight)),
            SafeRewardFunction(BrevityReward(weight=config.brevity_weight))
        ])


class HRMGRPOTrainer:
    """Enhanced GRPO trainer for HRM with curriculum and extensive logging"""
    
    def __init__(self, config: HRMGRPOConfig):
        self.config = config
        self.current_stage = "warmup"
        self.step_count = 0
        
        # Initialize W&B
        wandb.init(
            project="enhanced-hrm-grpo",
            name=f"hrm-{config.d_model}d-{config.max_cycles}c-grpo",
            config=config.__dict__
        )
        
        # Create model and tokenizer
        print("🚀 Creating HRM model and tokenizer...")
        self.model, self.tokenizer = create_hrm_model_and_tokenizer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_cycles=config.max_cycles,
            load_checkpoint="checkpoints/best_model.pt" if os.path.exists("checkpoints/best_model.pt") else None
        )
        
        # Create reference model for KL divergence
        print("📋 Creating reference model...")
        self.ref_model, _ = create_hrm_model_and_tokenizer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            max_cycles=config.max_cycles
        )
        
        # Copy weights to reference model
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
            
        print(f"🧠 Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.1f}M")
        
    def get_stage_and_rewards(self) -> Tuple[str, RewardCombiner]:
        """Get current stage and corresponding reward combiner"""
        
        if self.step_count < self.config.warmup_steps:
            stage = "warmup"
        elif self.step_count < self.config.warmup_steps + self.config.format_focus_steps:
            stage = "format_focus"
        else:
            stage = "reasoning"
            
        if stage != self.current_stage:
            print(f"🔄 Transitioning to stage: {stage}")
            self.current_stage = stage
            
        reward_combiner = create_adaptive_reward_combiner(stage, self.config)
        return stage, reward_combiner
        
    def compute_rewards(self, prompts: List[str], responses: List[str], **kwargs) -> Tuple[List[float], Dict]:
        """Compute rewards with extensive logging"""
        
        stage, reward_combiner = self.get_stage_and_rewards()
        
        # Extract references from kwargs
        references = [kwargs.get('reference_answer', [''])[i % len(kwargs.get('reference_answer', ['']))] 
                     for i in range(len(responses))]
        
        # Compute rewards
        total_rewards, component_rewards = reward_combiner.compute_rewards(
            responses, references, **kwargs
        )
        
        # Log reward statistics
        reward_stats = {
            f"reward/{reward_combiner.name}_mean": np.mean(total_rewards),
            f"reward/{reward_combiner.name}_std": np.std(total_rewards),
            f"reward/{reward_combiner.name}_min": np.min(total_rewards),
            f"reward/{reward_combiner.name}_max": np.max(total_rewards),
            "stage": stage,
            "step": self.step_count
        }
        
        # Log component rewards
        for component_name, component_values in component_rewards.items():
            reward_stats[f"reward_component/{component_name}_mean"] = np.mean(component_values)
            reward_stats[f"reward_component/{component_name}_std"] = np.std(component_values)
            
        # Log response quality metrics
        response_lengths = [len(r.split()) for r in responses]
        has_thinking = [bool(re.search(r'<think>', r)) for r in responses]
        has_solution = [bool(re.search(r'<SOLUTION>', r)) for r in responses]
        
        quality_stats = {
            "response/avg_length": np.mean(response_lengths),
            "response/thinking_rate": np.mean(has_thinking),
            "response/solution_rate": np.mean(has_solution),
            "response/empty_rate": np.mean([len(r.strip()) < 10 for r in responses])
        }
        
        reward_stats.update(quality_stats)
        
        # Log to W&B
        wandb.log(reward_stats)
        
        return total_rewards, reward_stats
        
    def train_stage(self, stage: str) -> None:
        """Train a specific stage of the curriculum"""
        
        print(f"🏃‍♂️ Training stage: {stage}")
        
        # Create dataset for this stage
        dataset = create_curriculum_dataset(self.config, stage)
        print(f"📊 Dataset size: {len(dataset)}")
        
        # Sample some examples to show
        for i in range(min(3, len(dataset))):
            example = dataset[i]
            print(f"Example {i+1}: {example['prompt'][:100]}...")
            
        # GRPO training arguments
        training_args = GRPOConfig(
            output_dir=f"./grpo_checkpoints/{stage}",
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            logging_steps=10,
            save_steps=100,
            eval_steps=100,
            warmup_steps=50,
            num_train_epochs=1 if stage == "warmup" else 2,
            report_to="wandb",
            remove_unused_columns=False,
            
            # GRPO specific
            loss_type=self.config.loss_type,
            kl_coeff=self.config.kl_coeff,
            num_generations=self.config.num_generations,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        
        # Create trainer
        trainer = GRPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.tokenizer,
            reward_function=self.compute_rewards,
        )
        
        # Train
        print(f"🚀 Starting GRPO training for {stage}...")
        trainer.train()
        
        # Update step count
        steps_this_stage = {
            "warmup": self.config.warmup_steps,
            "format_focus": self.config.format_focus_steps,
            "reasoning": self.config.reasoning_steps
        }
        
        self.step_count += steps_this_stage.get(stage, 500)
        
        # Save checkpoint
        checkpoint_path = f"grpo_checkpoints/{stage}_final"
        os.makedirs(checkpoint_path, exist_ok=True)
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        print(f"✅ Stage {stage} completed and saved!")
        
    def full_curriculum_training(self):
        """Run the complete curriculum training"""
        
        print("🎓 Starting full curriculum GRPO training!")
        print(f"📈 Curriculum: warmup({self.config.warmup_steps}) -> format_focus({self.config.format_focus_steps}) -> reasoning({self.config.reasoning_steps})")
        
        stages = ["warmup", "format_focus", "reasoning"]
        
        for stage in stages:
            self.train_stage(stage)
            
            # Test generation after each stage
            self.test_generation(stage)
            
        print("🎉 Full curriculum training completed!")
        
        # Final save
        final_path = "grpo_checkpoints/final_enhanced_hrm"
        os.makedirs(final_path, exist_ok=True)
        self.model.save_pretrained(final_path)
        self.tokenizer.save_pretrained(final_path)
        
        wandb.finish()
        
    def test_generation(self, stage: str):
        """Test generation quality after training stage"""
        
        test_prompts = [
            "What is 25 + 17?",
            "Solve step by step: If John has 5 apples and gives away 2, how many does he have?",
            "Write a function that finds the maximum of two numbers:",
            "What is the capital of Italy?"
        ]
        
        print(f"\n🎯 Testing generation after {stage}:")
        
        for prompt in test_prompts[:2]:  # Test 2 prompts
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + 100,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            generated_text = self.tokenizer.decode(generated[0], skip_special_tokens=False)
            response = generated_text[len(prompt):].strip()
            
            print(f"📝 Prompt: {prompt}")
            print(f"🤖 Response: {response[:200]}...")
            print("---")


def main():
    """Main training function"""
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Configuration
    config = HRMGRPOConfig(
        # Model settings
        d_model=256,
        n_heads=4,
        max_cycles=6,
        
        # Training settings
        num_generations=8,
        batch_size=4,
        learning_rate=5e-5,
        
        # Curriculum settings
        warmup_steps=100,      # Short warmup to get out of gibberish
        format_focus_steps=300, # Focus on learning format
        reasoning_steps=500,    # Full reasoning
        
        # Start with high format weight for gibberish models
        format_weight=3.0,
        correctness_weight=1.0
    )
    
    # Create trainer
    trainer = HRMGRPOTrainer(config)
    
    # Run full curriculum
    trainer.full_curriculum_training()


if __name__ == "__main__":
    main()