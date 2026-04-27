import torch
import numpy as np
from torch.utils.data import Dataset
from activations.cache_utils import load_activation_tensor

class ActivationDataset(Dataset):
    """
    Standard PyTorch Dataset to wrap extracted activations and their labels.
    """
    def __init__(self, activations: np.ndarray, labels: np.ndarray):
        self.activations = torch.from_numpy(activations).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.activations)

    def __getitem__(self, idx):
        return self.activations[idx], self.labels[idx]

def prepare_probing_data(activations_storage, layer_idx, act_type):
    X, y = [], []
    target_storage = activations_storage[act_type]

    for inst_id in target_storage["not_hallucinated"]:
        act = target_storage["not_hallucinated"][inst_id][layer_idx]
        X.append(act.detach().cpu().squeeze().numpy())
        y.append(0)

    for inst_id in target_storage["hallucinated"]:
        act = target_storage["hallucinated"][inst_id][layer_idx]
        X.append(act.detach().cpu().squeeze().numpy())
        y.append(1)

    return np.array(X), np.array(y)

# Example utility for later use
def get_probe_path(model_name, layer_idx, act_type, config):
    model_folder = model_name.split('/')[-1]
    return os.path.join(config.MODELS_DIR, model_folder, f"probe_L{layer_idx}_{act_type}.pt")