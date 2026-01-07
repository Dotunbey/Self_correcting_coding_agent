import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm

# Import your custom modules
from docker_env import CodeExecutionEnv
from rewards import RewardCalculator

# --- Configuration ---
BASE_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "my_coding_agent_v1" # Folder where PPO saved the model
NUM_TEST_SAMPLES = 10 # Keep small for CPU evaluation
MAX_GEN_LEN = 128

def get_test_data(n=10):
    """Fetches unseen problems from the APPS test split."""
    print("📥 Loading Test Data...")
    dataset = load_dataset("codeparrot/apps", split="test", streaming=True)
    test_samples = []
    
    # Iterate and filter for valid tests (similar to Phase 1)
    count = 0
    for item in dataset:
        try:
            import json
            io = json.loads(item["input_output"])
            if io.get("inputs") and io.get("outputs"):
                test_samples.append({
                    "question": item["question"],
                    "test_input": io["inputs"][0], # Use first test case
                    # Simple assertion logic for demo
                    "test_code": f"print('{io['inputs'][0]}')" 
                })
                count += 1
                if count >= n: break
        except:
            continue
    return test_samples

def generate_solution(model, tokenizer, question):
    """Generates code using the provided model."""
    prompt = f"<|im_start|>user\nWrite a Python solution for this:\n{question}\n<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=MAX_GEN_LEN, 
            do_sample=False, # Deterministic for eval
            temperature=0.0  # Greedy decoding
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract code block
    if "assistant\n" in response:
        response = response.split("assistant\n")[1]
    if "```python" in response:
        response = response.split("```python")[1].split("```")[0]
    return response.strip()

def evaluate():
    # 1. Setup Environment
    env = CodeExecutionEnv(timeout=3)
    calc = RewardCalculator()
    
    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)

    # 3. Load Base Model
    print("🏗️ Loading Base Model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME, 
        device_map="cpu", 
        torch_dtype=torch.float32
    )

    # 4. Load Test Data
    samples = get_test_data(NUM_TEST_SAMPLES)
    
    results = {"base": [], "rl": []}

    # 5. Run Evaluation Loop
    print(f"\n🚀 Evaluating on {NUM_TEST_SAMPLES} unseen problems...")
    
    for i, sample in enumerate(tqdm(samples)):
        question = sample["question"]
        test_code = sample["test_code"]

        # --- Evaluate Base Model ---
        code_base = generate_solution(base_model, tokenizer, question)
        out_base, pass_base = env.run(code_base, test_code)
        score_base = calc.compute(code_base, out_base, pass_base)
        results["base"].append(score_base)

    # 6. Switch to RL Model (Load Adapter)
    print("\n⚡ Loading RL Adapter...")
    # This mounts your trained weights ON TOP of the base model
    rl_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    for i, sample in enumerate(tqdm(samples)):
        question = sample["question"]
        test_code = sample["test_code"]

        # --- Evaluate RL Model ---
        code_rl = generate_solution(rl_model, tokenizer, question)
        out_rl, pass_rl = env.run(code_rl, test_code)
        score_rl = calc.compute(code_rl, out_rl, pass_rl)
        results["rl"].append(score_rl)

    # 7. Print Final Stats
    avg_base = sum(results["base"]) / len(results["base"])
    avg_rl = sum(results["rl"]) / len(results["rl"])
    
    print("\n" + "="*30)
    print("📊 EVALUATION RESULTS")
    print("="*30)
    print(f"Base Model Avg Reward: {avg_base:.2f}")
    print(f"RL Model Avg Reward:   {avg_rl:.2f}")
    print("-" * 30)
    if avg_rl > avg_base:
        print(f"✅ SUCCESS: Improvement of {((avg_rl - avg_base)/abs(avg_base or 1))*100:.1f}%")
    else:
        print("⚠️ RESULT: No significant improvement (Try training longer)")

if __name__ == "__main__":
    evaluate()
