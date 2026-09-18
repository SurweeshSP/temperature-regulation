from datasets import load_dataset
from typing import Dict, Any, List

from .gsm8k import process_gsm8k_example
from .bbh import process_bbh_example

def load_and_prepare_dataset(config: Dict[str, Any], smoke_test: bool = False) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads requested datasets and maps them to the canonical internal representation.
    Respects original splits.
    
    Returns:
        Dict mapping "dataset_task_split" -> List of canonical examples
    """
    all_data = {}
    
    # 1. GSM8K
    gsm8k_config = config.get('datasets', {}).get('gsm8k', {})
    if gsm8k_config.get('enabled', False):
        for sub_config in gsm8k_config.get('configs', []):
            try:
                ds = load_dataset("openai/gsm8k", sub_config)
                for split in ds.keys():
                    key = f"gsm8k_{sub_config}_{split}"
                    all_data[key] = []
                    
                    limit = 5 if smoke_test else len(ds[split])
                    for i, example in enumerate(ds[split]):
                        if i >= limit: break
                        all_data[key].append(process_gsm8k_example(example, sub_config))
            except Exception as e:
                print(f"Failed to load GSM8K {sub_config}: {e}")
                
    # 2. BBH
    bbh_config = config.get('datasets', {}).get('bbh', {})
    if bbh_config.get('enabled', False):
        for task in bbh_config.get('tasks', []):
            try:
                ds = load_dataset("maveriq/bigbenchhard", task, trust_remote_code=True)
                for split in ds.keys():
                    key = f"bbh_{task}_{split}"
                    all_data[key] = []
                    
                    limit = 5 if smoke_test else len(ds[split])
                    for i, example in enumerate(ds[split]):
                        if i >= limit: break
                        all_data[key].append(process_bbh_example(example, task))
            except Exception as e:
                print(f"Failed to load BBH {task}: {e}")
                
    return all_data
