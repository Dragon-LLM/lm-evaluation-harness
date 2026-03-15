import numpy as np
from lm_eval.api.task import ConfigurableTask
from lm_eval.api.instance import Instance

class PMIConfigurableTask(ConfigurableTask):
    def __init__(self, config=None, **kwargs):
        super().__init__(config=config, **kwargs)

    def construct_requests(self, doc, ctx, **kwargs):
        # Construction standard (gérée par le parent)
        standard_requests = super().construct_requests(doc, ctx, **kwargs)
        
        if not isinstance(standard_requests, list):
            standard_requests = [standard_requests]
            
        final_requests = []
        for req in standard_requests:
            final_requests.append(req) # Requete Standard (avec contexte)
            
            # Requete PMI (Copie avec contexte VIDE)
            new_args = req.arguments
            if isinstance(new_args, tuple) and len(new_args) >= 2:
                uncond_args = ("", new_args[1])
            else:
                uncond_args = new_args

            req_unconditional = Instance(
                request_type=req.request_type,
                doc=req.doc,
                arguments=uncond_args,
                idx=req.idx,
                metadata=req.metadata
            )
            final_requests.append(req_unconditional)
            
        return final_requests

    def process_results(self, doc, results):
        # 1. RECUPERATION DE LA CIBLE (GOLD)
        try:
            gold = self.doc_to_target(doc)
        except:
            gold = doc.get("gold", doc.get("label", doc.get("answer", doc.get("correct_answer_num"))))

        if isinstance(gold, str):
            if gold.isdigit():
                gold = int(gold)
            elif len(gold) == 1: 
                gold = ord(gold.upper()) - 65
        
        # 2. RECUPERATION DES CHOIX (POUR LA NORMALISATION)
        choices = []
        # Méthode officielle
        try:
            choices = self.doc_to_choice(doc)
        except:
            pass

        # Fallback manuel si la méthode officielle échoue (Belebele, XCSQA)
        if not choices:
            if "mc_answer1" in doc: # Belebele
                choices = [doc.get(f"mc_answer{i}") for i in range(1, 5) if f"mc_answer{i}" in doc]
            elif "question" in doc and "choices" in doc["question"] and "text" in doc["question"]["choices"]: # XCSQA
                choices = doc["question"]["choices"]["text"]
            elif "endings" in doc: # Hellaswag
                choices = doc["endings"]
            else: # MMLU
                choices = doc.get("choices", doc.get("options", []))

        # 3. CALCUL DES LONGUEURS
        if choices and isinstance(choices, list):
            try:
                # Max(1.0, ...) évite la division par zéro
                lengths = np.array([max(1.0, len(str(c).encode('utf-8'))) for c in choices])
            except:
                lengths = np.ones(len(choices))
        else:
            lengths = np.ones(len(results) // 2)

        # 4. CALCUL DU PMI
        # On récupère les log-probs
        cond_scores = [res[0] for res in results[::2]]   # P(Y|X)
        uncond_scores = [res[0] for res in results[1::2]] # P(Y)
        
        # Formule PMI : Log P(Y|X) - Log P(Y)
        pmi_scores = np.array(cond_scores) - np.array(uncond_scores)
        
        # PMI Brut (Celui qu'on mettra dans 'acc_pmi')
        pred_pmi = np.argmax(pmi_scores)
        
        # PMI Normalisé (Celui qu'on mettra dans 'acc_pmi_norm')
        # On divise le score PMI par la longueur de la réponse
        if len(lengths) == len(pmi_scores):
            pmi_norm_scores = pmi_scores / lengths
            pred_pmi_norm = np.argmax(pmi_norm_scores)
        else:
            pred_pmi_norm = pred_pmi

        return {
            "acc": 1.0 if pred_pmi == gold else 0.0,
            "acc_norm": 1.0 if pred_pmi_norm == gold else 0.0
        }