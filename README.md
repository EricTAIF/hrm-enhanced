# 🧠 Enhanced Hierarchical Reasoning Model (Enhanced HRM)

**A next-generation reasoning model that combines dual-channel thinking with GRPO training for structured problem-solving**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.56+-yellow.svg)](https://huggingface.co/transformers)
[![Weights & Biases](https://img.shields.io/badge/W%26B-Tracking-orange.svg)](https://wandb.ai)

## 🌟 Features

- **🎯 Dual Reasoning Channels**: Explicit token reasoning + latent thought processing
- **🔄 Dynamic Cycle Gating**: Entropy-based reasoning depth control
- **🎓 GRPO Training**: Group Relative Policy Optimization with comprehensive reward system
- **📚 Multi-Stage Curriculum**: Format learning → Math reasoning → Advanced thinking
- **⚡ HuggingFace Compatible**: Full integration with transformers ecosystem
- **📊 Comprehensive Metrics**: Real-time W&B tracking with detailed analytics

## 🏗️ Architecture

### Core Components

```
Enhanced HRM Architecture
├── 🧠 Planner Network (4 layers)
│   ├── Multi-head attention with RoPE
│   ├── Cycle-aware positional encoding
│   └── Entropy gating mechanism
├── 🔧 Worker Network (6 layers) 
│   ├── Reasoning block processing
│   ├── Cross-cycle skip connections
│   └── Consistency loss computation
└── 🎯 Dual-Head Output
    ├── Next-token prediction head
    └── Latent thought prediction head
```

### Key Innovations

- **Dynamic Reasoning Cycles**: Up to 4 adaptive reasoning steps
- **RoPE with Cycle Awareness**: Position encoding that understands reasoning depth
- **Cross-Cycle Consistency**: Skip connections between reasoning cycles
- **Thought Dimension**: 256-dim latent reasoning space

## 🚀 Quick Start

### Installation

```bash
git clone <repository>
cd enhanced_hrm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

# Install dependencies
pip install torch transformers trl datasets wandb safetensors
```

### Basic Usage

```python
from hrm_hf_wrapper import create_hrm_model_and_tokenizer

# Load Enhanced HRM
model, tokenizer = create_hrm_model_and_tokenizer(
    d_model=256,
    n_heads=4,
    max_cycles=4
)

# Generate with reasoning
prompt = "What is 15 + 27?"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_cycles=4, temperature=0.7)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## 🎓 Training

### Full GRPO Training Pipeline

```bash
# Extended overnight training (10-12 hours)
python working_grpo.py
```

### Training Stages

| Stage | Focus | Duration | Epochs | Dataset Size |
|-------|-------|----------|--------|--------------|
| **Stage 1** | Format Learning | ~30 min | 2 | 200 examples |
| **Stage 2** | Math Reasoning | ~2 hours | 5 | 500 examples |
| **Stage 3** | Advanced Reasoning | ~1.5 hours | 4 | 500 examples |
| **Stage 4** | Code Reasoning | ~1.5 hours | 3 | 300 examples |
| **Stage 5** | Chain-of-Thought | ~1.5 hours | 3 | 400 examples |

### Resume Training

The training automatically detects completed stages:

```bash
# Automatically resumes from latest checkpoint
python working_grpo.py
```

## 📊 Monitoring

### Weights & Biases Integration

Monitor training in real-time: [W&B Dashboard](https://wandb.ai/bergvall-eric/enhanced-hrm-grpo-full)

Key metrics to watch:
- **Loss**: GRPO policy loss (negative = improvement!)
- **Rewards**: Format, math, brevity components
- **Generation Quality**: Length, structure, coherence
- **Learning Dynamics**: Gradient norms, entropy, convergence

### Testing Checkpoints

```bash
# Test current model state
python test_generation.py
```

## 🏆 Expected Performance

### Learning Progression

```
Training Evolution:
Gibberish → Token Patterns → Structured Output → Mathematical Reasoning → Advanced Thinking

Example Outputs by Stage:
├── Initial: "千 construct Marshall svě награ..."
├── Stage 1: "logging officials compiling similarity..."  
├── Stage 2: "<think>I need to add 15 and 27</think><SOLUTION>42</SOLUTION>"
└── Stage 5: "<think>Let me break this down step by step...</think>..."
```

### Reward System

- **Format Reward**: Encourages `<think>` and `<SOLUTION>` structure
- **Math Reward**: Rewards correct numerical answers
- **Brevity Reward**: Prevents verbose, repetitive responses
- **Code Reward**: Promotes clean, functional code generation
- **Chain-of-Thought**: Rewards step-by-step reasoning

## 📁 Project Structure

```
enhanced_hrm/
├── README.md                    # This file
├── model.py                     # Core Enhanced HRM architecture  
├── hrm_hf_wrapper.py           # HuggingFace compatibility layer
├── modern_tokenizer.py         # Phi-3 tokenizer with reasoning tokens
├── reward_functions.py         # GRPO reward system
├── working_grpo.py             # Full training pipeline
├── test_generation.py          # Model testing utilities
├── grpo_stage_warmup/          # Stage 1 checkpoints
├── grpo_stage_math_focus/      # Stage 2 checkpoints
└── enhanced_hrm_grpo_final/    # Final trained model
```

## 🔧 Configuration

### Model Parameters

```python
config = {
    "d_model": 256,           # Hidden dimension
    "n_heads": 4,             # Attention heads
    "n_planner_layers": 4,    # Planner depth
    "n_worker_layers": 6,     # Worker depth  
    "max_cycles": 4,          # Max reasoning cycles
    "thought_dim": 256,       # Latent reasoning dimension
    "vocab_size": 32025,      # Including reasoning tokens
}
```

### Training Parameters

```python
grpo_config = {
    "learning_rate": 5e-5,
    "batch_size": 4,
    "gradient_accumulation_steps": 2,
    "num_generations": 8,
    "max_completion_length": 256,
    "temperature": 1.0,
}
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Memory Error**
   ```bash
   # Reduce batch size in working_grpo.py
   per_device_train_batch_size=2  # Instead of 4
   ```

2. **Checkpoint Loading Failed**
   ```python
   # The model uses custom architecture
   # Use safetensors loading as shown in test_generation.py
   ```

3. **Generation Device Mismatch** 
   ```python
   # Ensure model and inputs on same device
   model = model.to(device)
   inputs = {k: v.to(device) for k, v in inputs.items()}
   ```

## 📈 Performance Tips

### For Better Results

- **Longer Training**: Use extended configuration for 10+ hours
- **Larger Datasets**: Increase examples per stage
- **Lower Learning Rate**: More stable but slower convergence
- **More Cycles**: Increase `max_cycles` for deeper reasoning

### Resource Requirements

- **Minimum**: 8GB GPU (RTX 3070+)
- **Recommended**: 16GB+ GPU (RTX 4080+) 
- **Training Time**: 2-12 hours depending on configuration
- **Final Model Size**: ~67MB (16.7M parameters)

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Original HRM**: Hierarchical Reasoning Model concept
- **GRPO**: Group Relative Policy Optimization from TRL
- **Phi-3**: Microsoft's modern tokenizer architecture
- **Claude**: AI assistance in development and optimization

## 📚 Citation

```bibtex
@misc{enhanced_hrm_2025,
  title={Enhanced Hierarchical Reasoning Model with GRPO Training},
  author={Enhanced HRM Team},
  year={2025},
  note={A dual-channel reasoning model with comprehensive GRPO curriculum}
}
```

---

**🧠 "Making machines think step by step, one reasoning cycle at a time"** ✨# hrm-enhanced
