# export HF_ALLOW_CODE_EVAL=1

import os
import argparse
import logging
import json

from lm_eval import simple_evaluate, tasks
from lm_eval.utils import make_table
import bpb_task

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class BPBTaskManager(tasks.TaskManager):
    def load_task_or_group(self, task_list):
        bpb_requests = [t for t in task_list if t.endswith("_bpb")]
        standard_requests = [t for t in task_list if not t.endswith("_bpb")]

        loaded_tasks = {}

        if standard_requests:
            loaded_tasks.update(super().load_task_or_group(standard_requests))

        for bpb_name in bpb_requests:
            base_name = bpb_name[:-4]

            try:
                base_loaded = super().load_task_or_group([base_name])
                if base_name not in base_loaded:
                    logger.warning(f"Base task '{base_name}' not found.")
                    continue

                base_instance = base_loaded[base_name]

                if not hasattr(base_instance, "config"):
                    logger.warning(f"Base task '{base_name}' has no config.")
                    continue

                raw_config = base_instance.config
                config = (
                    raw_config.to_dict()
                    if hasattr(raw_config, "to_dict")
                    else vars(raw_config).copy()
                )

                config["task"] = bpb_name

                if "dataset_path" not in config and "path" in config:
                    config["dataset_path"] = config["path"]

                # Force BPB-safe evaluation behavior
                config["output_type"] = "loglikelihood"

                # Remove inherited generation / code-eval settings that break BPB
                for key in [
                    "unsafe_code",
                    "generation_kwargs",
                    "stop_sequence",
                    "fewshot_config",
                ]:
                    config.pop(key, None)

                # Replace inherited metrics entirely
                config["metric_list"] = [
                    {
                        "metric": "bpb",
                        "aggregation": "mean",
                        "higher_is_better": False,
                    }
                ]

                loaded_tasks[bpb_name] = bpb_task.BPBConfigurableTask(config=config)

                if not hasattr(self, "task_index"):
                    self.task_index = {}

                self.task_index[bpb_name] = {
                    "yaml_path": f"generated_{bpb_name}",
                    "group": None,
                    "task": bpb_name,
                }

                logger.info(
                    f"Loaded BPB wrapper task '{bpb_name}' from base task '{base_name}'"
                )

            except Exception as e:
                logger.error(f"Error initializing {bpb_name}: {e}")
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

    task_names = [t.strip() for t in args.tasks.split(",") if t.strip()]
    limit = None if args.limit is None else float(args.limit)

    logger.info(f"Starting evaluation on {args.device} with {args.num_fewshot} shot(s)")
    logger.info(f"Tasks: {task_names}")

    results = simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        task_manager=BPBTaskManager(),
        device=args.device,
        batch_size=args.batch_size,
        random_seed=args.seed,
        log_samples=args.log_samples,
        num_fewshot=args.num_fewshot,
        limit=limit,
    )

    print(make_table(results))

    if "results" in results:
        for task_name, metrics in results["results"].items():
            logger.info(f"Task '{task_name}' metrics: {list(metrics.keys())}")

    if args.output_path:
        if args.output_path.endswith(".json"):
            dir_to_create = os.path.dirname(args.output_path)
            file_path = args.output_path
        else:
            dir_to_create = args.output_path
            file_path = os.path.join(args.output_path, "results.json")

        if dir_to_create:
            os.makedirs(dir_to_create, exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"Results saved to {file_path}")


if __name__ == "__main__":
    main()


