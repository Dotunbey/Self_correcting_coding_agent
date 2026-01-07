import json
from datasets import load_dataset

def load_and_filter_apps():
    print("🚀 Downloading APPS dataset...")
    # Load the training split of the APPS dataset
    dataset = load_dataset("codeparrot/apps", split="train")
    
    print(f"✅ Original Size: {len(dataset)} problems")
    
    valid_problems = []
    
    print("🔍 Filtering for problems with unit tests...")
    for item in dataset:
        try:
            # The input_output field is a stringified JSON. We parse it.
            io = json.loads(item["input_output"])
            
            # We strictly need BOTH inputs and outputs to be present and non-empty
            if io.get("inputs") and io.get("outputs") and len(io["inputs"]) > 0:
                valid_problems.append({
                    "problem_id": item["problem_id"],
                    "question": item["question"],
                    "inputs": io["inputs"],
                    "outputs": io["outputs"],
                    "difficulty": item["difficulty"]
                })
        except json.JSONDecodeError:
            continue
            
    print(f"🎯 Filtered Size: {len(valid_problems)} valid problems ready for RL.")
    
    # Save to a local file for easier loading later
    with open("phase1_dataset.json", "w") as f:
        json.dump(valid_problems, f, indent=2)
        
    print("💾 Saved to 'phase1_dataset.json'")

if __name__ == "__main__":
    load_and_filter_apps()
