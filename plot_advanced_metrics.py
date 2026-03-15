import os
import json
import re
import numpy as np
import matplotlib.pyplot as plt
import csv
from collections import defaultdict
from scipy.stats import spearmanr, kendalltau
import glob

# --- CONFIGURATION ---
RESULTS_DIR = "results"
OUTPUT_DIR = os.path.join(RESULTS_DIR, "plots")
SUMMARY_FILE = os.path.join(OUTPUT_DIR, "models_comparison.csv")

# Baseline aléatoire (0.25 pour 4 choix)
RANDOM_BASELINE = 0.25 

# On ignore le début de l'entraînement pour la consistance (souvent bruyant)
# Mettre 0 si vos runs sont courts. HF utilise 15 milliards.
CONSISTENCY_START_TOKENS = 0 

TARGET_METRICS = {
    "acc,none": "Accuracy",
    "acc_norm,none": "Acc Norm"
}

def parse_token_count(step_str):
    step_str = step_str.upper()
    match = re.search(r'(\d+)(B|M|K)', step_str)
    if not match:
        match_digit = re.search(r'(\d+)', step_str)
        return int(match_digit.group(1)) if match_digit else 0
    number = int(match.group(1))
    unit = match.group(2)
    multipliers = {'B': 10**9, 'M': 10**6, 'K': 10**3}
    return number * multipliers.get(unit, 1)

def calculate_dataset_consistency(task_data, all_models, sorted_steps):
    """
    Calcule la 'Model Ordering Consistency' (Kendall's Tau).
    C'est un chiffre unique pour tout le graphique (stabilité du classement).
    """
    taus = []
    valid_steps = [s for s in sorted_steps if s >= CONSISTENCY_START_TOKENS]
    
    if len(valid_steps) < 2:
        return 0.0

    for i in range(len(valid_steps) - 1):
        step_t = valid_steps[i]
        step_t1 = valid_steps[i+1]
        
        scores_t = []
        scores_t1 = []
        
        for model in all_models:
            # On prend les scores seulement si le modèle existe à ces étapes
            val_t = task_data.get(model, {}).get(step_t)
            val_t1 = task_data.get(model, {}).get(step_t1)
            
            if val_t is not None and val_t1 is not None:
                scores_t.append(val_t)
                scores_t1.append(val_t1)
        
        # Il faut au moins 2 modèles pour avoir un classement
        if len(scores_t) >= 2:
            tau, _ = kendalltau(scores_t, scores_t1)
            if not np.isnan(tau):
                taus.append(tau)
            
    if not taus:
        return 0.0
        
    return np.mean(taus)

def calculate_individual_metrics(x_vals, y_vals):
    """
    Calcule les métriques propres à CHAQUE modèle.
    """
    # 1. Monotonicity (Spearman Rank)
    if len(x_vals) < 2: return 0, 0, 0
    mono, _ = spearmanr(x_vals, y_vals)
    
    # 2. SNR (Approximation temporelle)
    deltas = np.diff(y_vals)
    if np.std(deltas) == 0:
        snr = 0.0
    else:
        snr = np.abs(np.mean(deltas)) / np.std(deltas)
        
    # 3. Non-Randomness (Max - Baseline)
    max_score = np.max(y_vals)
    non_randomness = max_score - RANDOM_BASELINE
    
    return mono, snr, non_randomness

def main():
    raw_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    
    print(f"🔍 Lecture des fichiers dans {RESULTS_DIR}...")
    json_files = glob.glob(os.path.join(RESULTS_DIR, "**", "*.json"), recursive=True)
    
    all_steps = set()
    all_tasks = set()
    all_models = set()

    # 1. Extraction
    for filepath in json_files:
        try:
            path_parts = os.path.relpath(filepath, RESULTS_DIR).split(os.sep)
            if len(path_parts) < 3: continue
            
            model_name = path_parts[0]
            step_raw = path_parts[1]
            tokens = parse_token_count(step_raw)
            if tokens == 0: continue
            
            all_steps.add(tokens)
            all_models.add(model_name)

            with open(filepath, 'r') as f:
                content = json.load(f)
            
            if "results" not in content: continue

            for task_key, task_res in content["results"].items():
                all_tasks.add(task_key)
                for json_key, human_name in TARGET_METRICS.items():
                    if json_key in task_res:
                        raw_data[model_name][human_name][task_key][tokens] = task_res[json_key]
        except: continue

    sorted_steps = sorted(list(all_steps))
    sorted_models = sorted(list(all_models))
    
    # 2. Génération
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    csv_rows = []
    # Consistency est propre au dataset, mais on la répète pour chaque ligne pour faciliter l'analyse
    headers = ["Dataset", "Metric", "Dataset_Consistency", "Model", "Monotonicity", "SNR", "Non_Randomness"]
    csv_rows.append(headers)

    print(f"\n{'='*110}")
    print(f"{'DATASET / MODÈLE':<40} | {'MONO':<6} | {'SNR':<6} | {'NON-RND':<8} | {'GLOBAL CONS':<10}")
    print(f"{'='*110}")

    for metric_human in TARGET_METRICS.values():
        for task_name in sorted(list(all_tasks)):
            
            # --- Préparation des données pour ce Dataset ---
            task_models_data = {} # {model: {step: score}}
            has_data = False
            for m in sorted_models:
                if task_name in raw_data[m][metric_human]:
                    task_models_data[m] = raw_data[m][metric_human][task_name]
                    has_data = True
            
            if not has_data: continue

            # --- A. CALCUL DE LA CONSISTENCY DU DATASET (Unique par graphe) ---
            global_consistency = calculate_dataset_consistency(task_models_data, sorted_models, sorted_steps)

            # --- B. PLOT ---
            plt.figure(figsize=(14, 9))
            ax = plt.subplot(111)
            
            for model in sorted_models:
                step_dict = task_models_data.get(model)
                if not step_dict: continue

                x_vals, y_vals = [], []
                for idx, step in enumerate(sorted_steps):
                    if step in step_dict:
                        x_vals.append(step)
                        y_vals.append(step_dict[step])
                
                if len(x_vals) < 2: continue

                # Calcul des métriques individuelles
                mono, snr, non_rnd = calculate_individual_metrics(x_vals, y_vals)
                
                # CSV
                csv_rows.append([task_name, metric_human, f"{global_consistency:.3f}", model, 
                                 f"{mono:.3f}", f"{snr:.3f}", f"{non_rnd:.3f}"])
                
                # Console
                display_name = f"{task_name[:15]}.. - {model[:20]}"
                print(f"{display_name:<40} | {mono:6.2f} | {snr:6.2f} | {non_rnd:8.3f} | {global_consistency:6.3f}")

                # Plot
                x_display = [x/1e9 for x in x_vals]
                # Légende : Juste les métriques du modèle. La consistency est dans le titre.
                label = f"{model} (M:{mono:.2f} S:{snr:.1f} NR:{non_rnd:.2f})"
                ax.plot(x_display, y_vals, marker='o', label=label)

            # Titre incluant la CONSISTENCY globale
            plt.title(f"Dataset: {task_name} ({metric_human})\nDataset Consistency (Ranking Stability): {global_consistency:.2f}")
            plt.xlabel("Entraînement (Milliards de Tokens)")
            plt.ylabel(f"Score ({metric_human})")
            plt.axhline(y=RANDOM_BASELINE, color='r', ls='--', alpha=0.3, label="Hasard (0.25)")
            
            ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Modèles")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            safe_task = task_name.replace("/", "_")
            safe_metric = metric_human.replace(" ", "_")
            safe_name = os.path.join(OUTPUT_DIR, f"{safe_task}_{safe_metric}.png")
            plt.savefig(safe_name, bbox_inches='tight')
            plt.close()

    with open(SUMMARY_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)
    
    print(f"\n✅ Terminé. Graphiques sauvegardés dans {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()