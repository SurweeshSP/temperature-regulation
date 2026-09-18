import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def compute_auroc(y_true, y_scores):
    try:
        return roc_auc_score(y_true, y_scores)
    except ValueError:
        return float('nan') # Handle single class

def compute_auprc(y_true, y_scores):
    try:
        return average_precision_score(y_true, y_scores)
    except ValueError:
        return float('nan')

def compute_mae(y_true, y_pred):
    return np.mean(np.abs(np.array(y_true) - np.array(y_pred)))

def compute_rmse(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def compute_temperature_diagnostics(temperatures):
    t = np.array(temperatures)
    return {
        "mean": float(np.mean(t)),
        "std": float(np.std(t)),
        "min": float(np.min(t)),
        "max": float(np.max(t)),
        "cv": float(np.std(t) / (np.mean(t) + 1e-8)),
        "pct_at_min": float(np.mean(t == np.min(t))),
        "pct_at_max": float(np.mean(t == np.max(t)))
    }
