from typing import Dict, Any

def normalize_bbh_answer(answer: str, task: str) -> str:
    """
    Normalizes BBH answers based on the specific task.
    """
    if not isinstance(answer, str):
        return str(answer)
        
    ans = answer.strip().lower()
    
    if task == "boolean_expressions":
        if "true" in ans:
            return "true"
        if "false" in ans:
            return "false"
            
    elif task == "causal_judgement":
        if "yes" in ans:
            return "yes"
        if "no" in ans:
            return "no"
            
    elif task == "date_understanding":
        # Usually multiple choice or exact date strings. For robust MC extraction:
        # It's usually " (A)", " (B)" etc.
        import re
        match = re.search(r'\(([A-F])\)', answer)
        if match:
            return match.group(1).upper()
            
    # Default fallback: return stripped string
    return ans

def evaluate_bbh_answer(prediction: str, target: str, task: str) -> bool:
    """
    Deterministically evaluates if the prediction matches the target for BBH.
    """
    pred_norm = normalize_bbh_answer(prediction, task)
    targ_norm = normalize_bbh_answer(target, task)
    return pred_norm == targ_norm

def process_bbh_example(example: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """
    Maps a raw BBH example into the canonical internal format.
    """
    return {
        "dataset": "bbh",
        "task": task_name,
        "id": hash(example.get('input', '')),
        "prompt": example.get('input', ''),
        "target": example.get('target', ''),
        "metadata": {}
    }
