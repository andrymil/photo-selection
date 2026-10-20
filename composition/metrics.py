import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(predictions, targets):
    return {
        "mse": float(np.mean((predictions - targets) ** 2)),
        "srcc": float(spearmanr(predictions, targets).statistic),
        "plcc": float(pearsonr(predictions, targets).statistic),
    }
