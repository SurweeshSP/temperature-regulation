import torch
from typing import Optional
from .confidence import compute_confidence
from .entropy import compute_entropy, compute_normalized_entropy
from .uncertainty import compute_uncertainty
from .novelty import NoveltyReference
from .conflict import compute_conflict
from .stability import compute_stability
from ..metacognition.meta_state import MetaState
from ..base.interface import TransformerState

class StateExtractor:
    def __init__(self, vocab_size: int, config: dict):
        self.vocab_size = vocab_size
        self.config = config
        self.novelty_ref = NoveltyReference(method=config.get("novelty", "cosine"))
        
        self.h_prev = None
        self.p_prev = None
        
    def reset_history(self):
        self.h_prev = None
        self.p_prev = None

    def __call__(self, state: TransformerState, alt_p: Optional[torch.Tensor] = None) -> MetaState:
        confidence = compute_confidence(state.probabilities)
        
        entropy = compute_entropy(state.logits)
        normalized_entropy = compute_normalized_entropy(entropy, self.vocab_size)
        
        uncertainty = compute_uncertainty(
            estimator_type=self.config.get("uncertainty", "normalized_entropy"),
            normalized_entropy=normalized_entropy
        )
        
        if self.novelty_ref.mu_ref is not None:
            novelty = self.novelty_ref.score(state.hidden_state)
        else:
            novelty = torch.zeros_like(confidence)
            
        conflict = compute_conflict(state.probabilities, alt_p)
        
        stability = compute_stability(
            state.hidden_state, self.h_prev,
            state.probabilities, self.p_prev
        )
        
        # Store for next step
        self.h_prev = state.hidden_state.detach()
        self.p_prev = state.probabilities.detach()
        
        return MetaState(
            confidence=confidence,
            uncertainty=uncertainty,
            novelty=novelty,
            conflict=conflict,
            stability=stability,
            entropy=entropy,
            normalized_entropy=normalized_entropy
        )
