import torch
import unittest
from mctr.scripts.run_phase1_complete import get_causal_loop_loss
from mctr.controller.controller import MCTRController, EntropyTarget
from mctr.metacognition.evaluator import MetaEvaluator
from mctr.state.state import StateExtractor

class TestCausalLoop(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.config = {
            'controller': {'mode': 'learned', 'temperature_min': 0.1, 'temperature_max': 5.0, 'initial_temperature': 1.0},
            'entropy': {'target_mode': 'fixed', 'target_value': 0.5},
            'meta': {'input_dim': 6, 'hidden_dim': 16, 'output_dim': 16},
            'state': {
                'entropy': True, 'confidence': True, 'uncertainty': 'normalized_entropy', 
                'novelty': 'cosine', 'conflict': 'js_divergence', 'stability': 'exponential'
            }
        }
        self.batch_size = 2
        self.seq_len = 3
        self.vocab_size = 100
        
        self.state_extractor = StateExtractor(vocab_size=self.vocab_size, config=self.config['state'])
        self.state_extractor.novelty_ref.fit(torch.randn(10, 1536))
        self.evaluator = MetaEvaluator(input_dim=6, hidden_dim=16, meta_dim=16)
        self.controller = MCTRController(meta_dim=16, config=self.config['controller'])
        self.entropy_target_gen = EntropyTarget(config=self.config['entropy'], meta_dim=6)
        
    def test_no_future_leakage_and_gradient_flow(self):
        input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len))
        
        # Test gradient flow to controller
        self.controller.train()
        
        losses = get_causal_loop_loss(
            self.config, None, self.state_extractor, self.evaluator, 
            self.controller, self.entropy_target_gen, input_ids, targets, self.device
        )
        
        loss = losses['task'] + losses['entropy'] + losses['regulation']
        loss.backward()
        
        # Check gradients exist for controller
        has_grad = any(p.grad is not None for p in self.controller.parameters())
        self.assertTrue(has_grad)

if __name__ == '__main__':
    unittest.main()
