import os
import json
import config


def save_layer_metrics(dataset_name, activation_type, layer_idx, metrics_dict):
    """
    Saves metrics into a structured JSON format inside the runs/ directory.
    Follows a strict hierarchy: runs/results/MODEL/DATASET/ACTIVATION_TYPE/metrics_layerX.json
    """
    model_safe_name = config.MODEL_ID.split("/")[-1]

    # Define the directory path
    target_dir = os.path.join(
        config.RESULTS_DIR,
        model_safe_name,
        dataset_name,
        activation_type
    )
    os.makedirs(target_dir, exist_ok = True)

    file_path = os.path.join(target_dir, f"metrics_layer{layer_idx}.json")

    with open(file_path, "w") as f:
        json.dump(metrics_dict, f, indent = 4)