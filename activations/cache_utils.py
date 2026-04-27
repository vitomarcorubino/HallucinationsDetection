import os
import torch
import re
from typing import List, Dict, Union
import config


def save_activation_tensor(tensor: torch.Tensor, layer_idx: Union[int, str], instance_id: int,
                           act_type: str, dataset_name: str, label_name: str, config_obj):
    """Saves a single activation tensor to a structured directory."""
    model_safe_name = config_obj.MODEL_ID.split('/')[-1]
    path = os.path.join(
        config_obj.BASE_PROJECT_DIR,
        "activation_cache",
        model_safe_name,
        dataset_name,
        f"activation_{act_type}",
        label_name
    )
    os.makedirs(path, exist_ok = True)

    file_name = f"layer{layer_idx}-id{instance_id}.pt"
    torch.save(tensor.float(), os.path.join(path, file_name))


def load_activation_tensor(layer_idx: int, instance_id: int, act_type: str,
                           dataset_name: str, label_name: str, config_obj):
    """Loads a specific tensor from disk."""
    model_safe_name = config_obj.MODEL_ID.split('/')[-1]
    path = os.path.join(
        config_obj.BASE_PROJECT_DIR, "activation_cache",
        model_safe_name, dataset_name, f"activation_{act_type}", label_name,
        f"layer{layer_idx}-id{instance_id}.pt"
    )
    return torch.load(path, weights_only = True)


# NEW: Aggiunto parametro check_only_dir
def check_activations_exist(ids: List[int], layers: List[int], act_types: List[str],
                            dataset_name: str, label_name: str, check_only_dir: bool = False) -> bool:
    """
    Verifies if activation files already exist on disk.
    If check_only_dir is True, it simply verifies the directory is present and not empty.
    """
    model_name = config.MODEL_ID.split('/')[-1]

    for act_type in act_types:
        path = os.path.join(
            config.BASE_PROJECT_DIR, "activation_cache", model_name,
            dataset_name, f"activation_{act_type}", label_name
        )

        if not os.path.exists(path):
            return False

        files_in_folder = set(os.listdir(path))

        # NEW: Fast check for cross-probing
        if check_only_dir:
            if len(files_in_folder) == 0:
                return False
            continue

        for idx in ids:
            for layer in layers:
                filename = f"layer{layer}-id{idx}.pt"
                if filename not in files_in_folder:
                    return False
    return True


# NEW: Aggiunto parametro load_all
def load_activations_for_group(ids: List[int], layers: List[int], act_types: List[str],
                               dataset_name: str, label_name: str, load_all: bool = False) -> Dict:
    """
    Loads saved .pt tensors into a nested dictionary structure.
    If load_all is True, it loads all files found in the directory dynamically.
    """
    model_name = config.MODEL_ID.split('/')[-1]
    group_data = {act_type: {} for act_type in act_types}

    for act_type in act_types:
        path = os.path.join(
            config.BASE_PROJECT_DIR, "activation_cache", model_name,
            dataset_name, f"activation_{act_type}", label_name
        )

        # NEW: Dynamic loading for cross-probing
        if load_all:
            if os.path.exists(path):
                for filename in os.listdir(path):
                    if filename.endswith(".pt"):
                        # Extract layer and id from 'layerX-idY.pt'
                        match = re.match(r"layer(\d+)-id(\d+)\.pt", filename)
                        if match:
                            layer = int(match.group(1))
                            idx = int(match.group(2))

                            if layer in layers:
                                if idx not in group_data[act_type]:
                                    group_data[act_type][idx] = {}
                                file_path = os.path.join(path, filename)
                                group_data[act_type][idx][layer] = torch.load(file_path, weights_only = True)
        else:
            for idx in ids:
                if idx not in group_data[act_type]:
                    group_data[act_type][idx] = {}
                for layer in layers:
                    file_path = os.path.join(path, f"layer{layer}-id{idx}.pt")
                    group_data[act_type][idx][layer] = torch.load(file_path, weights_only = True)

    return group_data