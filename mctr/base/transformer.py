import torch
from transformers import AutoModelForCausalLM
from typing import Optional
from .interface import TransformerState

class MCTRTransformerWrapper:
    def __init__(self, model_name: str, device: str = "auto", dtype=torch.bfloat16):
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=dtype,
            output_hidden_states=True,
            output_attentions=True
        )
        self.model.eval()
        
    @torch.no_grad()
    def forward_with_state(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, return_attention: bool = False) -> TransformerState:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=return_attention
        )
        
        hidden_states = outputs.hidden_states[-1]
        logits = outputs.logits
        
        log_probs = torch.log_softmax(logits, dim=-1)
        probabilities = log_probs.exp()
        
        attention_maps = outputs.attentions if return_attention else None
        
        return TransformerState(
            hidden_state=hidden_states,
            logits=logits,
            probabilities=probabilities,
            attention_maps=attention_maps
        )
