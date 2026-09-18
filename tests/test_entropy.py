import torch
import math
from mctr.state.entropy import compute_entropy, compute_normalized_entropy

def test_entropy_bounds():
    vocab_size = 1000
    batch_size = 4
    seq_len = 10
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    entropy = compute_entropy(logits)
    
    # 0 <= H <= log(|V|)
    assert torch.all(entropy >= 0)
    assert torch.all(entropy <= math.log(vocab_size) + 1e-5)
    
    # Check normalized
    norm_entropy = compute_normalized_entropy(entropy, vocab_size)
    assert torch.all(norm_entropy >= 0)
    assert torch.all(norm_entropy <= 1.0 + 1e-5)
    
    # Check deterministic
    logits_2 = logits.clone()
    entropy_2 = compute_entropy(logits_2)
    assert torch.allclose(entropy, entropy_2)
    
    # Check stable probabilities (sum to 1)
    probs = torch.log_softmax(logits, dim=-1).exp()
    assert torch.allclose(probs.sum(dim=-1), torch.ones_like(probs.sum(dim=-1)), atol=1e-5)
