from dataclasses import dataclass
from typing import Optional, Tuple
import torch
from torch import Tensor

@dataclass
class TransformerState:
    hidden_state: Tensor
    logits: Tensor
    probabilities: Tensor
    attention_maps: Optional[Tuple[Tensor, ...]] = None
