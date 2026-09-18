import torch
import unittest
from mctr.controller.controller import MCTRController, EntropyTarget

class TestMCTRController(unittest.TestCase):
    def setUp(self):
        self.meta_dim = 32
        self.batch_size = 4
        self.config_learned = {
            'mode': 'learned',
            'temperature_min': 0.1,
            'temperature_max': 5.0,
            'hidden_dim': 16
        }
        self.config_feedback = {
            'mode': 'entropy_feedback',
            'temperature_min': 0.1,
            'temperature_max': 5.0,
            'eta_T': 0.1
        }
        
    def test_learned_mode_shape_and_bounds(self):
        controller = MCTRController(self.meta_dim, self.config_learned)
        m_t = torch.randn(self.batch_size, self.meta_dim)
        
        control_state = controller(m_t)
        T_t = control_state.temperature
        
        self.assertEqual(T_t.shape, (self.batch_size,))
        self.assertTrue(torch.all(T_t >= self.config_learned['temperature_min']))
        self.assertTrue(torch.all(T_t <= self.config_learned['temperature_max']))
        
    def test_feedback_mode(self):
        controller = MCTRController(self.meta_dim, self.config_feedback)
        m_t = torch.randn(self.batch_size, self.meta_dim)
        current_T = torch.ones(self.batch_size) * 1.0
        current_H = torch.ones(self.batch_size) * 0.8
        target_H = torch.ones(self.batch_size) * 0.5 # delta = -0.3
        
        control_state = controller(m_t, current_T=current_T, current_H=current_H, target_H=target_H)
        T_t = control_state.temperature
        
        # T_{t+1} = T_t + eta * (H* - H) = 1.0 + 0.1 * (-0.3) = 0.97
        expected = torch.ones(self.batch_size) * 0.97
        self.assertTrue(torch.allclose(T_t, expected))
        
    def test_entropy_target_fixed(self):
        config = {'target_mode': 'fixed', 'target_value': 0.6}
        target_gen = EntropyTarget(config, meta_dim=6)
        
        psi_t = torch.randn(self.batch_size, 6)
        H_star = target_gen(psi_t)
        
        self.assertEqual(H_star.shape, (self.batch_size,))
        self.assertTrue(torch.allclose(H_star, torch.ones(self.batch_size) * 0.6))
        
    def test_entropy_target_adaptive(self):
        config = {'target_mode': 'adaptive', 'target_hidden_dim': 16}
        target_gen = EntropyTarget(config, meta_dim=6)
        
        psi_t = torch.randn(self.batch_size, 6)
        H_star = target_gen(psi_t)
        
        self.assertEqual(H_star.shape, (self.batch_size,))
        self.assertTrue(torch.all(H_star >= 0.0))
        self.assertTrue(torch.all(H_star <= 1.0))

if __name__ == '__main__':
    unittest.main()
