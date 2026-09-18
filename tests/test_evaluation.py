import unittest
import numpy as np
from mctr.evaluation.metrics import compute_auroc, compute_auprc, compute_temperature_diagnostics
from mctr.evaluation.calibration import compute_ece, compute_brier_score

class TestEvaluationMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = [0, 1, 1, 0, 1]
        self.y_prob = [0.1, 0.9, 0.8, 0.4, 0.85]
        
    def test_auroc_auprc(self):
        auroc = compute_auroc(self.y_true, self.y_prob)
        auprc = compute_auprc(self.y_true, self.y_prob)
        
        self.assertTrue(auroc > 0.9)
        self.assertTrue(auprc > 0.9)
        
    def test_ece(self):
        ece = compute_ece(self.y_true, self.y_prob, n_bins=5)
        self.assertTrue(isinstance(ece, float))
        self.assertTrue(ece >= 0.0)
        
    def test_brier(self):
        brier = compute_brier_score(self.y_true, self.y_prob)
        self.assertTrue(isinstance(brier, float))
        self.assertTrue(brier >= 0.0)
        
    def test_temperature_diagnostics(self):
        temps = [0.5, 0.5, 1.0, 1.0, 1.5]
        diags = compute_temperature_diagnostics(temps)
        
        self.assertAlmostEqual(diags['mean'], 0.9)
        self.assertEqual(diags['min'], 0.5)
        self.assertEqual(diags['max'], 1.5)
        self.assertAlmostEqual(diags['pct_at_min'], 0.4)
        self.assertAlmostEqual(diags['pct_at_max'], 0.2)

if __name__ == '__main__':
    unittest.main()
