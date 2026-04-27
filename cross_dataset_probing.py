import os
import argparse
import torch
import json
import joblib  
import numpy as np # <-- Aggiunto
import config
from probing.model import LogisticRegressionProbe, FFNProbe 
from probing.data_prep import prepare_probing_data
from activations.layer_configs import get_layer_map
from activations.cache_utils import load_activations_for_group, check_activations_exist
from utils.evaluation import compute_probe_metrics
from utils.plotting import plot_hallucination_metrics 

def run_cross_evaluation(source_model_id, target_datasets):
    print(f"\n{'=' * 50}")
    print(f"🔄 CROSS-DATASET EVALUATION (Multi-Prober & Multi-Run)")
    print(f"Trained on: NQ-Swap")
    print(f"Testing on: {', '.join(target_datasets)}")
    print(f"{'=' * 50}\n")

    layer_map = get_layer_map(config.MODEL_ID, config.TARGET_LAYERS)
    act_types = list(layer_map.keys())
    model_folder = source_model_id.split('/')[-1]

    # 1. Load and Merge Target Dataset Activations (Test Set)
    combined_activations = {at: {"not_hallucinated": {}, "hallucinated": {}} for at in act_types}

    for dataset_arg in target_datasets:
        if dataset_arg.startswith("ragbench_"):
            dataset_id = dataset_arg
            ds_folder = dataset_arg.replace("ragbench_", "")
        else:
            dataset_id = f"ragbench_{dataset_arg}"
            ds_folder = dataset_arg
        
        config.BASE_PROJECT_DIR = os.path.join(config.RUNS_DIR, ds_folder)
        config.ACTIVATION_CACHE_DIR = os.path.join(config.BASE_PROJECT_DIR, "activation_cache")
        
        print(f"📥 Loading test activations from: {ds_folder}...")
        for label_name in ["not_hallucinated", "hallucinated"]:
            if not check_activations_exist([], config.TARGET_LAYERS, act_types, dataset_id, label_name, check_only_dir=True):
                print(f"❌ Error: Activations for {dataset_id} ({label_name}) not found.")
                return

            loaded_group = load_activations_for_group([], config.TARGET_LAYERS, act_types, dataset_id, label_name, load_all=True)
            for at in act_types:
                for k, v in loaded_group[at].items():
                    unique_id = f"{ds_folder}_{k}"
                    combined_activations[at][label_name][unique_id] = v

    # 2. Setup Evaluation Parameters
    if len(target_datasets) == 1:
        combined_folder = target_datasets[0].replace("ragbench_", "")
        safe_plot_name = f"cross_{combined_folder}"
    else:
        combined_folder = "ragbench_combined"
        safe_plot_name = "cross_ragbench_combined"
    
    config.BASE_PROJECT_DIR = os.path.join(config.RUNS_DIR, combined_folder)
    base_cross_results_dir = os.path.join(config.BASE_PROJECT_DIR, "results_cross")
    
    probe_types = ["logistic_regression", "ffn", "svm"]
    num_runs = 5

    # 3. Triple Loop: Probe Type -> Activation -> Layer -> Run
    for p_type in probe_types:
        print(f"\n--- Evaluating Cross-Dataset Performance: {p_type.upper()} ---")
        p_type_dir = os.path.join(base_cross_results_dir, p_type)
        
        for act_type in act_types:
            act_dir = os.path.join(p_type_dir, act_type)
            os.makedirs(act_dir, exist_ok=True)
            
            for layer_idx in config.TARGET_LAYERS:
                X_test, y_test = prepare_probing_data(combined_activations, layer_idx, act_type)
                if len(X_test) == 0: continue

                # NEW: Initialize dictionary to collect metrics across the 5 runs
                layer_metrics_history = {"ACC": [], "F1": [], "AUC": [], "AUPRC": []}

                for run_idx in range(num_runs):
                    probe_base_path = os.path.join(config.MODELS_DIR, model_folder, p_type, f"probe_L{layer_idx}_{act_type}_run{run_idx}")
                    
                    # --- Loading Logic ---
                    if p_type == "svm":
                        full_path = f"{probe_base_path}.joblib"
                        if not os.path.exists(full_path): continue
                        probe = joblib.load(full_path)
                    else:
                        full_path = f"{probe_base_path}.pt"
                        if not os.path.exists(full_path): continue
                        
                        input_dim = X_test.shape[1]
                        if p_type == "logistic_regression":
                            probe = LogisticRegressionProbe(input_dim).to(config.DEVICE)
                        else:
                            probe = FFNProbe(input_dim).to(config.DEVICE)
                        
                        probe.load_state_dict(torch.load(full_path, map_location=config.DEVICE))

                    # --- Metrics Computation ---
                    metrics = compute_probe_metrics(probe, X_test, y_test, config.DEVICE)
                    
                    # NEW: Append to history tracker
                    for metric_name, value in metrics.items():
                        layer_metrics_history[metric_name].append(value)
                    
                    # Save metrics with run suffix
                    with open(os.path.join(act_dir, f"metrics_layer{layer_idx}_run{run_idx}.json"), "w") as f:
                        json.dump(metrics, f)
                
                # NEW: Calculate Mean and Standard Deviation and save summary
                if layer_metrics_history["ACC"]: # Evita errori se non ha trovato modelli
                    summary_metrics = {}
                    for metric_name, values in layer_metrics_history.items():
                        summary_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                        summary_metrics[f"{metric_name}_std"] = float(np.std(values))
                    
                    summary_filename = f"metrics_layer{layer_idx}_summary.json"
                    with open(os.path.join(act_dir, summary_filename), "w") as f:
                        json.dump(summary_metrics, f, indent=4)
                
                print(f"  Layer {layer_idx} {act_type} completed (5 runs + summary).")

    # 4. Generate Plots
    original_results_dir = getattr(config, 'RESULTS_DIR', None)
    config.IMAGES_DIR = os.path.join(config.BASE_PROJECT_DIR, "images")

    for p_type in probe_types:
        config.RESULTS_DIR = os.path.join(base_cross_results_dir, p_type)
        print(f"\n📊 Generating Cross-Dataset plots for {p_type.upper()}...")
        plot_hallucination_metrics(f"{safe_plot_name}_{p_type}")
    
    if original_results_dir:
        config.RESULTS_DIR = original_results_dir

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_datasets", nargs='+', required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        config.DEVICE = "cpu"

    run_cross_evaluation(config.MODEL_ID, args.target_datasets)