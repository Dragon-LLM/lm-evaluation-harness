import subprocess
import os
import json
import re

# Tâche BPB
TASKS = ["mbpp_bpb"] #humaneval
TASKS_STRING = ",".join(TASKS)

MODELS = [
    "/e/project1/jureap140/7BA1B/converted_hf/dragon-1B-dragon_nemotron_code",
    "/e/project1/jureap140/7BA1B/converted_hf/dragon-1B-dragon_nemotron_code_v1",
    "/e/project1/jureap140/7BA1B/converted_hf/dragon-1B-dragon_nemotron_code_v2",
    "/e/project1/jureap140/7BA1B/converted_hf/dragon-1B-dragon_stack_code_v2" 
]

SEEDS = [0]
OUTPUT_DIR = "results_bpb_code_fs_5"
LIMIT = 50
BATCH_SIZE = 1
NUM_FEWSHOT = 5


def find_iter_checkpoints(model_path):
    """
    Trouve tous les sous-dossiers iter_xxxx dans le chemin donné.
    Retourne une liste triée par numéro d'itération.
    """
    if not os.path.isdir(model_path):
        return []

    iter_dirs = []
    for item in os.listdir(model_path):
        if re.match(r"^iter_\d+$", item):
            full_path = os.path.join(model_path, item)
            if os.path.isdir(full_path):
                iter_dirs.append(full_path)

    iter_dirs.sort(key=lambda x: int(re.search(r"iter_(\d+)", x).group(1)))
    return iter_dirs


def already_done(json_file):
    """
    Vérifie si le run semble déjà terminé proprement.
    """
    if not os.path.exists(json_file):
        return False

    try:
        with open(json_file, "r") as f:
            data = json.load(f)

        if "results" not in data:
            return False

        if "mbpp_bpb" not in data["results"]:
            return False

        # On considère le run fini si la tâche est bien présente
        return True
    except Exception:
        return False


def run_evaluation():
    print(f"Tasks to run: {TASKS_STRING}")

    for model in MODELS:
        is_local = os.path.isdir(model)

        if not is_local:
            print(f"Skipping non-local model path: {model}")
            continue

        iter_checkpoints = find_iter_checkpoints(model)

        if iter_checkpoints:
            model_paths = iter_checkpoints
            base_model_name = os.path.basename(model.rstrip("/"))
            print(f"\n{'=' * 60}")
            print(f"Local Model with checkpoints: {model}")
            print(f"Found {len(model_paths)} iter checkpoints")
            print(f"{'=' * 60}")
        else:
            model_paths = [model]
            base_model_name = os.path.basename(os.path.dirname(model.rstrip("/")))
            print(f"\n{'=' * 60}")
            print(f"Local Model: {model}")
            print(f"{'=' * 60}")

        for model_path in model_paths:
            iter_name = os.path.basename(model_path.rstrip("/"))

            for seed in SEEDS:
                output_path = os.path.join(
                    OUTPUT_DIR, base_model_name, iter_name, f"seed_{seed}"
                )
                json_file = os.path.join(output_path, "results.json")

                if already_done(json_file):
                    print(f"Skipping {base_model_name}/{iter_name} (already completed)")
                    continue

                print(f"Processing: {base_model_name}/{iter_name} | Seed {seed}")

                model_args = (
                    f"pretrained={model_path},"
                    f"trust_remote_code=True"
                )

                cmd = [
                    "python",
                    "bpb_runner.py",
                    "--model",
                    "hf",
                    "--model_args",
                    model_args,
                    "--tasks",
                    TASKS_STRING,
                    "--device",
                    "cuda:0",
                    "--batch_size",
                    str(BATCH_SIZE),
                    "--limit",
                    str(LIMIT),
                    "--output_path",
                    output_path,
                    "--log_samples",
                    "--num_fewshot",
                    str(NUM_FEWSHOT),
                    "--seed",
                    str(seed),
                    "--confirm_run_unsafe_code",
                ]

                env = os.environ.copy()
                env["HF_ALLOW_CODE_EVAL"] = "1"

                try:
                    subprocess.run(cmd, check=True, env=env)
                except subprocess.CalledProcessError as e:
                    print(f"Error processing {base_model_name}/{iter_name}: {e}")


if __name__ == "__main__":
    run_evaluation()