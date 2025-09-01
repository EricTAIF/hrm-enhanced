"""
Quick test of Enhanced HRM generation during GRPO training
"""

import torch
from hrm_hf_wrapper import create_hrm_model_and_tokenizer

def test_current_model():
    print("🧪 Testing Enhanced HRM Generation...")
    
    # Load model and tokenizer
    model, tokenizer = create_hrm_model_and_tokenizer(
        d_model=256,
        n_heads=4, 
        max_cycles=4
    )
    
    # Ensure model is on the right device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"📍 Model moved to: {device}")
    
    # Try to load the latest checkpoint if available
    try:
        checkpoint_path = "./grpo_stage_warmup/checkpoint-100"  # Final Stage 1 checkpoint
        print(f"📂 Attempting to load checkpoint: {checkpoint_path}")
        
        # Load safetensors weights directly
        from safetensors import safe_open
        
        tensors = {}
        with safe_open(f"{checkpoint_path}/model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
        
        # Load the state dict, handling missing/extra keys gracefully
        missing_keys, unexpected_keys = model.load_state_dict(tensors, strict=False)
        
        print("✅ Checkpoint loaded from safetensors!")
        if missing_keys:
            print(f"📝 Missing keys (normal): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"📝 Unexpected keys: {len(unexpected_keys)} keys")
        
    except Exception as e:
        print(f"⚠️ Checkpoint loading failed: {e}")
        print("📝 Using fresh model state (this is normal early in training)")
    
    model.eval()
    
    # Test prompts
    test_prompts = [
        "What is 5 + 3?",
        "Solve step by step: 12 - 8",
        "Calculate: 3 × 4", 
        "Say hello",
        "What color is grass?"
    ]
    
    print("\n🎯 Enhanced HRM Generations (Current State):")
    print("=" * 60)
    
    for prompt in test_prompts:
        inputs = tokenizer(prompt, return_tensors="pt")
        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_length=inputs['input_ids'].shape[1] + 50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
                max_cycles=4
            )
        
        generated_text = tokenizer.decode(generated[0], skip_special_tokens=False)
        response = generated_text[len(prompt):].strip()
        
        print(f"\n📝 Input: '{prompt}'")
        print(f"🤖 Output: '{response[:120]}{'...' if len(response) > 120 else ''}'")
        print("-" * 40)

if __name__ == "__main__":
    test_current_model()