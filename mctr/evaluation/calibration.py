import numpy as np

def compute_ece(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        mask = binids == i
        if np.any(mask):
            prob_mean = np.mean(y_prob[mask])
            acc_mean = np.mean(y_true[mask])
            ece += (np.sum(mask) / len(y_prob)) * np.abs(prob_mean - acc_mean)
            
    return ece

def compute_brier_score(y_true, y_prob):
    """
    Brier Score.
    """
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    return np.mean((y_prob - y_true)**2)
