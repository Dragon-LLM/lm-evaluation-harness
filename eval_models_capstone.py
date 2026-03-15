# eval_models_capstone.py
import subprocess
from huggingface_hub import list_repo_refs

MODELS = [
    "HPLT/fineweb-2.1.0-fra_Latn-llama-2b-100bt",
    "HPLT/hplt-3.0-fra_Latn-llama-2b-100bt",
    "HPLT/hplt-3.0-fra_Latn-top-llama-2b-100bt",
    "HPLT/madlad-400-1.0-fra_Latn-llama-2b-100bt"
]

TASKS = "mlmm_mmlu_fr,mlmm_hellaswag_fr"
SEEDS = [0, 1, 2, 3, 4]

for model in MODELS:
    model_name = model.split("/")[-1]
    
    refs = list_repo_refs(model)
    revisions = [b.name for b in refs.branches]
    
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Found {len(revisions)} revisions: {revisions}")
    print('='*50)
    
    for rev in revisions:
        for seed in SEEDS:
            print(f"\n--- Evaluating: {model} @ {rev} | Seed {seed} ---")
            cmd = [
                "python", "-m", "lm_eval",
                "--model", "hf",
                "--model_args", f"pretrained={model},revision={rev}",
                "--tasks", TASKS,
                "--device", "cuda:0",
                "--batch_size", "4",
                "--seed", str(seed),
                "--output_path", f"results/{model_name}/{rev}/seed_{seed}"
            ]
            subprocess.run(cmd)