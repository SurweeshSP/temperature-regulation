import re
from typing import Dict, Any, Optional

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """
    Extracts the numerical answer from a GSM8K response.
    GSM8K answers are typically formatted with '#### ' preceding the final number.
    If '####' is not present, we attempt to find the last number in boxed formatting,
    or the last number in the text as a fallback.
    """
    if not isinstance(text, str):
        return None
        
    # Standard GSM8K format
    if "####" in text:
        ans = text.split("####")[-1].strip()
        # Remove commas and extract just the number part if there's trailing text
        ans = ans.replace(",", "")
        match = re.search(r'[-+]?\d*\.?\d+', ans)
        if match:
            return match.group()
            
    # Boxed format fallback
    boxed_matches = re.findall(r'\\boxed{(.*?)}', text)
    if boxed_matches:
        ans = boxed_matches[-1].strip().replace(",", "")
        match = re.search(r'[-+]?\d*\.?\d+', ans)
        if match:
            return match.group()
            
    # Absolute fallback: last number in the text
    text_no_commas = text.replace(",", "")
    matches = re.findall(r'[-+]?\d*\.?\d+', text_no_commas)
    if matches:
        return matches[-1]
        
    return None

def normalize_gsm8k_answer(answer: str) -> Optional[float]:
    """
    Normalizes the extracted answer to a float for robust equality checking.
    """
    try:
        if answer is None:
            return None
        return float(answer)
    except ValueError:
        return None

def process_gsm8k_example(example: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    """
    Maps a raw GSM8K example into the canonical internal format.
    """
    question = example.get('question', '')
    raw_answer = example.get('answer', '')
    
    extracted_target = extract_gsm8k_answer(raw_answer)
    
    return {
        "dataset": "gsm8k",
        "task": task_name,
        "id": example.get('id', hash(question)),
        "prompt": question,
        "target": extracted_target, # Extracted numerical string
        "raw_target": raw_answer,
        "metadata": {}
    }
