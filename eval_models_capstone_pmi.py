import subprocess
import os
import json
import copy
import re
from huggingface_hub import list_repo_refs

# LISTE DE VOS TÂCHES
BASE_TASKS = [
    "belebele_fr",
    # "belebele_fra_Latn",
    "hellaswag_fr",
    # "m_mmlu_fr",
    "mlmm_hellaswag_fr",
    "mlmm_mmlu_fr",
    "xcsqa_fr"
]

# On construit la liste complète : Tâche Normale + Tâche PMI
TASKS_TO_RUN = []
for t in BASE_TASKS:
    TASKS_TO_RUN.append(t)           # Pour acc, acc_norm
    TASKS_TO_RUN.append(f"{t}_pmi")  # Pour le PMI (stocké temporairement dans acc)

TASKS_STRING = ",".join(TASKS_TO_RUN)

MODELS = [
    "/e/project1/jureap140/7BA1B/converted_hf/dragon-1B-dragon_datamerge2",
    # "HPLT/fineweb-2.1.0-fra_Latn-llama-2b-100bt",
    # "HPLT/hplt-3.0-fra_Latn-llama-2b-100bt",
    # "HPLT/hplt-3.0-fra_Latn-top-llama-2b-100bt",
    # "HPLT/madlad-400-1.0-fra_Latn-llama-2b-100bt",
]

def find_iter_checkpoints(model_path):
    """
    Trouve tous les sous-dossiers iter_xxxx dans le chemin donné.
    Retourne une liste triée par numéro d'itération.
    """
    if not os.path.isdir(model_path):
        return []
    
    iter_dirs = []
    for item in os.listdir(model_path):
        if re.match(r'^iter_\d+$', item):
            full_path = os.path.join(model_path, item)
            if os.path.isdir(full_path):
                iter_dirs.append(full_path)
    
    # Trier par numéro d'itération
    iter_dirs.sort(key=lambda x: int(re.search(r'iter_(\d+)', x).group(1)))
    return iter_dirs

SEEDS = [0]
#OUTPUT_DIR = "results_merged_final"
# OUTPUT_DIR = "results_3_few_shots"
NUM_FEWSHOT = 0
OUTPUT_DIR = "results_fr"

def merge_pmi_results(filepath):
    """
    Fusionne les résultats PMI dans les tâches standards et renomme les clés.
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if "results" not in data:
            return

        results = data["results"]
        # On itère sur les tâches de base pour aller chercher leur équivalent PMI
        for task in BASE_TASKS:
            pmi_task = f"{task}_pmi"
            
            if task in results and pmi_task in results:
                # On récupère les scores PMI (qui sont cachés sous 'acc' et 'acc_norm')
                pmi_data = results[pmi_task]
                
                # On les injecte dans la tâche principale avec le BON NOM
                if "acc,none" in pmi_data:
                    results[task]["acc_pmi,none"] = pmi_data["acc,none"]
                if "acc_norm,none" in pmi_data:
                    results[task]["acc_pmi_norm,none"] = pmi_data["acc_norm,none"]
                if "acc_stderr,none" in pmi_data:
                    results[task]["acc_pmi_stderr,none"] = pmi_data["acc_stderr,none"]
                
                # On supprime l'entrée temporaire PMI pour nettoyer le fichier
                del results[pmi_task]
        
        # Sauvegarde du fichier propre
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  [Auto-Merge] Successfully merged PMI scores into standard tasks for {filepath}")

    except Exception as e:
        print(f"  [Error] Failed to merge JSON: {e}")

def run_evaluation():
    print(f"Tasks to run: {TASKS_STRING}")
    
    for model in MODELS:
        # Détecte si c'est un modèle local ou HuggingFace
        is_local = os.path.isdir(model)
        
        if is_local:
            # Cherche les sous-dossiers iter_xxxx
            iter_checkpoints = find_iter_checkpoints(model)
            
            if iter_checkpoints:
                # Si on trouve des iter_xxxx, on les utilise
                model_paths = iter_checkpoints
                base_model_name = os.path.basename(model.rstrip("/"))
                print(f"\n{'='*50}")
                print(f"Local Model with checkpoints: {model}")
                print(f"Found {len(model_paths)} iter checkpoints")
                print('='*50)
            else:
                # Sinon on utilise le modèle directement (cas iter_xxxx direct ou autre)
                model_paths = [model]
                base_model_name = os.path.basename(os.path.dirname(model.rstrip("/")))
                print(f"\n{'='*50}")
                print(f"Local Model: {model}")
                print('='*50)
            
            for model_path in model_paths:
                iter_name = os.path.basename(model_path.rstrip("/"))
                
                for seed in SEEDS:
                    output_path = os.path.join(OUTPUT_DIR, base_model_name, iter_name, f"seed_{seed}")
                    json_file = os.path.join(output_path, "results.json")
                    
                    # Vérification intelligente
                    if os.path.exists(json_file):
                        try:
                            with open(json_file) as f:
                                existing_data = json.load(f)
                            if "results" in existing_data and BASE_TASKS[0] in existing_data["results"]:
                                if "acc_pmi,none" in existing_data["results"][BASE_TASKS[0]]:
                                    print(f"Skipping {base_model_name}/{iter_name} (Already completed and merged)")
                                    continue
                        except:
                            pass
                    
                    print(f"Processing: {base_model_name}/{iter_name} | Seed {seed}")
                    
                    model_args = f"pretrained={model_path},dtype=bfloat16,trust_remote_code=True"
                    
                    cmd = [
                        "python", "pmi_runner.py", 
                        "--model", "hf",
                        "--model_args", model_args,
                        "--tasks", TASKS_STRING,
                        "--device", "cuda:0",
                        "--batch_size", "4",
                        "--seed", str(seed),
                        "--output_path", output_path,
                        "--num_fewshot", str(NUM_FEWSHOT)
                    ]

                    try:
                        subprocess.run(cmd, check=True)
                        merge_pmi_results(json_file)
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing {base_model_name}/{iter_name}: {e}")
        else:
            # Modèle HuggingFace
            try:
                print(f"Fetching refs for {model}...")
                refs = list_repo_refs(model)
                revisions = [b.name for b in refs.branches]
            except Exception as e:
                print(f"Error fetching refs: {e}")
                continue
            model_name = model.split("/")[-1]
            print(f"\n{'='*50}")
            print(f"Model: {model} ({len(revisions)} revisions)")
            print('='*50)
            
            for rev in revisions:
                for seed in SEEDS:
                    output_path = os.path.join(OUTPUT_DIR, model_name, rev, f"seed_{seed}")
                    json_file = os.path.join(output_path, "results.json")
                    
                    if os.path.exists(json_file):
                        try:
                            with open(json_file) as f:
                                existing_data = json.load(f)
                            if "results" in existing_data and BASE_TASKS[0] in existing_data["results"]:
                                if "acc_pmi,none" in existing_data["results"][BASE_TASKS[0]]:
                                    print(f"Skipping {model_name} {rev} (Already completed and merged)")
                                    continue
                        except:
                            pass

                    print(f"Processing: {model_name} @ {rev} | Seed {seed}")
                    
                    model_args = f"pretrained={model},revision={rev},dtype=bfloat16,trust_remote_code=True"
                    
                    cmd = [
                        "python", "pmi_runner.py", 
                        "--model", "hf",
                        "--model_args", model_args,
                        "--tasks", TASKS_STRING,
                        "--device", "cuda:0",
                        "--batch_size", "4",
                        "--seed", str(seed),
                        "--output_path", output_path,
                        "--num_fewshot", str(NUM_FEWSHOT)
                    ]

                    try:
                        subprocess.run(cmd, check=True)
                        merge_pmi_results(json_file)
                    except subprocess.CalledProcessError as e:
                        print(f"Error processing {model_name} at {rev}: {e}")

if __name__ == "__main__":
    run_evaluation()