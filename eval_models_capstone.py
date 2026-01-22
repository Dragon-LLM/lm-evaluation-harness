# eval_models.py

import subprocess
from huggingface_hub import list_repo_refs

MODELS = [
    "HPLT/fineweb-2.1.0-fra_Latn-llama-2b-100bt",
    "HPLT/hplt-3.0-fra_Latn-llama-2b-100bt",
    "HPLT/hplt-3.0-fra_Latn-top-llama-2b-100bt",
    "HPLT/madlad-400-1.0-fra_Latn-llama-2b-100bt"
]
TASKS = "belebele_fr,mlmm_hellaswag_fr,mlmm_mmlu_fr,xcsqa_fr,belebele_fra_Latn,hellaswag_fr,m_mmlu_fr"
#TASKS = "belebele_fr"
for model in MODELS:
    model_name = model.split("/")[-1]
    
    # Get all revisions for this model
    refs = list_repo_refs(model)
    revisions = [b.name for b in refs.branches]
    
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print(f"Found {len(revisions)} revisions: {revisions}")
    print('='*50)
    
    for rev in revisions:
        print(f"\n--- Evaluating: {model} @ {rev} ---")
        cmd = [
            "python", "-m", "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={model},revision={rev}",
            "--tasks", TASKS,
            "--device", "cuda:0",
            "--batch_size", "4",
            "--output_path", f"results/{model_name}/{rev}"
        ]
        subprocess.run(cmd)