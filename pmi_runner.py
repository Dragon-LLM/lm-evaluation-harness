import sys
import os
import argparse
import logging
import json
import torch
from lm_eval import simple_evaluate, tasks
from lm_eval.utils import make_table
import pmi_task

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

class PMITaskManager(tasks.TaskManager):
    def load_task_or_group(self, task_list):
        pmi_requests = [t for t in task_list if t.endswith('_pmi')]
        standard_requests = [t for t in task_list if not t.endswith('_pmi')]
        
        loaded_tasks = {}
        if standard_requests:
            loaded_tasks.update(super().load_task_or_group(standard_requests))

        for pmi_name in pmi_requests:
            base_name = pmi_name[:-4]
            try:
                base_loaded = super().load_task_or_group([base_name])
                if base_name not in base_loaded:
                    logger.warning(f"Base task '{base_name}' not found.")
                    continue
                
                base_instance = base_loaded[base_name]
                if hasattr(base_instance, 'config'):
                    raw_config = base_instance.config
                    config = raw_config.to_dict() if hasattr(raw_config, 'to_dict') else vars(raw_config).copy()

                    config['task'] = pmi_name 
                    if 'dataset_path' not in config and 'path' in config:
                        config['dataset_path'] = config['path']
                    
                    loaded_tasks[pmi_name] = pmi_task.PMIConfigurableTask(config=config)
                    
                    if not hasattr(self, 'task_index'): 
                        self.task_index = {}
                    self.task_index[pmi_name] = {
                        "yaml_path": f"generated_{pmi_name}", 
                        "group": None, 
                        "task": pmi_name
                    }
            except Exception as e:
                logger.error(f"Error initializing {pmi_name}: {e}")
                continue
        return loaded_tasks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--log_samples", action="store_true")
    parser.add_argument("--limit", default=None)
    parser.add_argument("--num_fewshot", type=int, default=0)
    
    args, _ = parser.parse_known_args()
    
    logger.info(f"Starting evaluation on {args.device} with {args.num_fewshot} shot(s)")
    
    results = simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=args.tasks.split(","),
        task_manager=PMITaskManager(),
        device=args.device,
        batch_size=args.batch_size,
        random_seed=args.seed,
        log_samples=args.log_samples,
        num_fewshot=args.num_fewshot,
        limit=float(args.limit) if args.limit else None
    )

    if args.output_path:
        if args.output_path.endswith('.json'):
            dir_to_create = os.path.dirname(args.output_path)
            file_path = args.output_path
        else:
            dir_to_create = args.output_path
            file_path = os.path.join(args.output_path, "results.json")
            
        os.makedirs(dir_to_create, exist_ok=True)
        # ---------------------------------------------
        
        print(make_table(results))
        
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Results saved to {file_path}")

if __name__ == "__main__":
    main()