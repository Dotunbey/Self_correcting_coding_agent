import torch
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from docker_env import CodeExecutionEnv
from rewards import RewardCalculator
import json

# --- Configuration ---
MODEL_NAME = "meta-llama/Meta-Llama-3-8B-Instruct"
DATA_FILE = "phase1_dataset.json" # Created in Phase 1

config = PPOConfig(
    model_name=MODEL_NAME,
    learning_rate=1.41e-5,
    batch_size=16,          # Global batch size
    mini_batch_size=4,      # Per-device batch size
    gradient_accumulation_steps=1,
    optimize_cuda_cache=True,
    target_kl=0.1,          # Prevent model mode collapse
    ppo_epochs=4,
    seed=42,
)

def build_dataset(tokenizer, data_path):
    """Loads and formats the dataset for TRL."""
    with open(data_path, "r") as f:
        raw_data = json.load(f)
    
    # Format prompts exactly like Llama-3 expects
    formatted_data = []
    for item in raw_data:
        prompt = (
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
            f"Write a Python function to solve this:\n{item['question']}\n"
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            "```python\n"
        )
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
        formatted_data.append({
            "query": prompt,
            "input_ids": tokens,
            "tests": item["inputs"] # Pass tests along for the env
        })
    
    # Return as a standard dataset object
    from datasets import Dataset
    return Dataset.from_list(formatted_data)

def main():
    # 1. Initialize Components
    env = CodeExecutionEnv()
    reward_calc = RewardCalculator()
    
    # Load Model with Value Head (Required for PPO)
    # Note: load_in_4bit requires 'bitsandbytes' for consumer GPU memory efficiency
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        config.model_name,
        load_in_4bit=True,
        device_map="auto",
        peft_config=None 
    )
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # 2. Prepare Data
    dataset = build_dataset(tokenizer, DATA_FILE)

    # 3. Setup Trainer
    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None, # TRL automatically creates a reference model to calc KL
        tokenizer=tokenizer,
        dataset=dataset,
    )

    print("🚀 Starting PPO Training Loop...")

    # 4. Training Loop
    for epoch, batch in enumerate(ppo_trainer.dataloader):
        query_tensors = batch["input_ids"]
        
        # A. Generate Code (Action)
        # We sample from the model policy
        response_tensors = ppo_trainer.generate(
            query_tensors,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            pad_token_id=tokenizer.eos_token_id
        )
        
        batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
        
        # B. Execute & Reward (Environment)
        rewards = []
        for i, code in enumerate(batch["response"]):
            # Extract just the python code block
            clean_code = code.split("```python")[-1].split("```")[0]
            
            # Construct unit tests from dataset inputs
            # (Simplified: In prod, you'd iterate through inputs/outputs)
            test_cases = "" # Placeholder: Add logic to convert batch['tests'][i] to assert strings
            
            logs, passed = env.run(clean_code, test_cases)
            reward = reward_calc.compute(clean_code, logs, passed)
            rewards.append(torch.tensor(reward))

        # C. Optimization Step (PPO)
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Logging
        mean_reward = torch.stack(rewards).mean()
        print(f"Step {epoch}: Mean Reward: {mean_reward:.4f}")
        ppo_trainer.log_stats(stats, batch, rewards)

    # 5. Save Final Model
    ppo_trainer.save_pretrained("self-correcting-coder-v1")
    print("✅ Training Complete. Model Saved.")

if __name__ == "__main__":
    main()
