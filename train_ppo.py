import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from peft import LoraConfig
from datasets import load_dataset

# Import your custom modules
from docker_env import CodeExecutionEnv
from rewards import RewardCalculator

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"  # Lightweight model for CPU usage
EPOCHS = 1
BATCH_SIZE = 4  # Keep small for CPU
LR = 1.41e-5

def format_prompt(example):
    """Formats the APPS problem into a prompt Qwen understands."""
    prompt = f"<|im_start|>user\nWrite a Python solution for this:\n{example['question']}\n<|im_end|>\n<|im_start|>assistant\n"
    return {"query": prompt}

def main():
    # 1. Initialize Environment & Rewards
    env = CodeExecutionEnv(timeout=3)
    reward_calc = RewardCalculator()

    # 2. Setup LoRA (Low-Rank Adaptation)
    # This freezes the main model and trains a tiny adapter, saving massive RAM.
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 3. Load Model & Tokenizer
    # device_map="cpu" forces CPU usage.
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        MODEL_NAME,
        peft_config=peft_config,
        device_map="cpu",
        torch_dtype=torch.float32  # CPU works best with float32
    )
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 4. Load & Preprocess Data (from Phase 1)
    dataset = load_dataset("json", data_files="phase1_dataset.json", split="train")
    dataset = dataset.map(format_prompt)
    
    # Tokenize dataset
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["query"], truncation=True, max_length=512)
        return sample
    
    dataset = dataset.map(tokenize, batched=False)
    
    # 5. Initialize PPO Trainer
    config = PPOConfig(
        model_name=MODEL_NAME,
        learning_rate=LR,
        batch_size=BATCH_SIZE,
        mini_batch_size=BATCH_SIZE, # No micro-batching needed on CPU
        gradient_accumulation_steps=1,
    )

    ppo_trainer = PPOTrainer(
        config=config,
        model=model,
        ref_model=None, # None = shared weights (saves RAM)
        tokenizer=tokenizer,
        dataset=dataset,
        data_collator=lambda data: dict((key, [d[key] for d in data]) for key in data[0]),
    )

    # 6. The RL Loop
    print(f"🚀 Starting PPO Training on CPU using {MODEL_NAME}...")
    
    for epoch in range(EPOCHS):
        for i, batch in tqdm(enumerate(ppo_trainer.dataloader)):
            
            # A. Prepare Inputs
            query_tensors = [torch.tensor(t).long().to("cpu") for t in batch["input_ids"]]
            
            # B. Generate Code (The Policy Action)
            # We constrain generation to keep it short for speed
            response_tensors = ppo_trainer.generate(
                query_tensors, 
                max_new_tokens=128, 
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            
            batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
            
            # C. Interact with Environment (Docker)
            rewards = []
            for query, response, tests_json in zip(batch["query"], batch["response"], batch["inputs"]):
                # Clean response to extract just the python code
                # (Simple split to remove prompt, in production use regex)
                generated_code = response.replace(query, "").strip()
                if "```python" in generated_code:
                    generated_code = generated_code.split("```python")[1].split("```")[0]

                # Convert JSON inputs to python assert statements
                # Note: This uses the first test case from the dataset for the reward signal
                # Real implementation would loop through all tests.
                test_case = tests_json[0] 
                # Assuming simple string io for demo stability
                test_code = f"print('{test_case}')" 

                # Run in Docker
                output, passed = env.run(generated_code, test_code)
                
                # Calculate Reward
                score = reward_calc.compute(generated_code, output, passed, num_tests=1)
                rewards.append(torch.tensor(score, dtype=torch.float32))

            # D. PPO Update Step
            # This updates the model weights based on the rewards
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Logging
            print(f"Step {i}: Mean Reward = {torch.stack(rewards).mean().item():.2f}")

    # 7. Save the Adapter
    ppo_trainer.save_pretrained("my_coding_agent_v1")
    print("💾 Model saved locally.")

if __name__ == "__main__":
    main()
