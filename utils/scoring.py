import torch
import numpy as np
from scipy.stats import kurtosis # Necessario per la curtosi
from .text_helpers import normalize_answer


def exact_match_score(prediction: str, ground_truth: str) -> bool:
    """Checks if normalized prediction matches normalized ground truth."""
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """Computes the highest score among all available references."""
    scores_for_ground_truths = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores_for_ground_truths) if scores_for_ground_truths else 0


def check_grounding_in_raw_context(prediction: str, sub_context: str) -> int:
    """Verifies if the predicted answer is a substring of the context."""
    normalized_prediction = normalize_answer(prediction)
    normalized_context = normalize_answer(sub_context)
    return 1 if normalized_prediction in normalized_context else 0


def compute_activation_stats(activations: torch.Tensor):
    """
    Computes L1, L2, Kurtosis, Hoyer and Gini indices.
    Reshapes input to 2D [samples, neurons] if a 1D vector is received.
    """
    if torch.is_tensor(activations):
        act_np = activations.detach().cpu().numpy()
    else:
        act_np = np.array(activations)

    # FIX: Se l'array è 1D, lo trasformiamo in 2D [1, N]
    if act_np.ndim == 1:
        # Se è un singolo vettore, lo trattiamo come 1 campione
        # Se è un vettore lunghissimo (frutto di un cat errato), 
        # questa riga evita il crash dell'AxisError.
        act_np = act_np.reshape(1, -1)
    
    # 1. Norms
    l1 = np.linalg.norm(act_np, ord=1, axis=1).mean()
    l2 = np.linalg.norm(act_np, ord=2, axis=1).mean()
    
    # 2. Kurtosis
    kurt = kurtosis(act_np, axis=1).mean()
    
    # 3. Hoyer Index
    d = act_np.shape[1]
    l1_per_sample = np.linalg.norm(act_np, ord=1, axis=1)
    l2_per_sample = np.linalg.norm(act_np, ord=2, axis=1)
    l2_per_sample[l2_per_sample == 0] = 1e-9
    hoyer = ((np.sqrt(d) - (l1_per_sample / l2_per_sample)) / (np.sqrt(d) - 1)).mean()
    
    # 4. Gini Index
    def gini(array):
        array = np.abs(array)
        array = np.sort(array, axis=1)
        index = np.arange(1, array.shape[1] + 1)
        n = array.shape[1]
        return ((np.sum((2 * index - n - 1) * array, axis=1)) / (n * np.sum(array, axis=1)))

    gini_idx = gini(act_np).mean()
    
    return {
        "L1": float(l1), "L2": float(l2), "Kurtosis": float(kurt),
        "Hoyer": float(hoyer), "Gini": float(gini_idx)
    }