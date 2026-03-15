import subprocess
import os

# Comparing your HPLT run against specialized code models
MODELS = [
    "Qwen/Qwen2.5-Coder-1.5B",               # The current champion
    "bigcode/starcoder2-3b",                 # High-quality data baseline
    "deepseek-ai/deepseek-coder-1.3b-base"   # Efficiency baseline
]

# Focusing purely on code and general reasoning (no French)
TASKS = "humaneval,mbpp,Minerva,gsm8k"

for model in MODELS:
    model_name = model.split("/")[-1]
    
    # Define a simple "main" revision if not iterating through steps
    revisions = ["main"] 
    
    # If evaluating your HPLT model, you'd insert your 5B step logic here
    for rev in revisions:
        out_path = f"results/code_bench/{model_name}"
        os.makedirs(out_path, exist_ok=True)

        print(f"📡 Evaluating Code Specialist: {model_name}")
        
        cmd = [
            "lm-eval", "--model", "hf",
            "--model_args", f"pretrained={model},trust_remote_code=True",
            "--tasks", TASKS,
            "--device", "cuda:0",
            "--batch_size", "auto",
            "--output_path", out_path
        ]
        subprocess.run(cmd)