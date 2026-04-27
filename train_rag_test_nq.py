import os
import argparse
import torch
import json
import numpy as np # <-- Aggiunto
import config
from probing.model import LogisticRegressionProbe, FFNProbe
from probing.data_prep import prepare_probing_data
from probing.trainer import train_single_probe, save_probe
from activations.layer_configs import get_layer_map
from activations.cache_utils import load_activations_for_group, check_activations_exist
from utils.evaluation import compute_probe_metrics
from utils.plotting import plot_hallucination_metrics 
from sklearn.model_selection import train_test_split

def run_reverse_experiment(source_model_id, target_test_datasets):
    train_datasets = ["ragbench_covidqa", "ragbench_cuad", "ragbench_delucionqa"]
    
    print(f"\n{'=' * 50}")
    print(f"🔄 REVERSE CROSS-DATASET EVALUATION (Multi-Prober & Multi-Run)")
    print(f"Training on: {', '.join(train_datasets)} (Combined RAG)")
    print(f"Testing on: {', '.join(target_test_datasets)} (NQ-Swap)")
    print(f"{'=' * 50}\n")

    layer_map = get_layer_map(config.MODEL_ID, config.TARGET_LAYERS)
    act_types = list(layer_map.keys())
    model_folder_name = source_model_id.split('/')[-1]
    custom_model_name_for_saving = f"{model_folder_name}_trained_on_RAG"

    # --- 1. LOAD TRAINING DATA (COMBINED RAGBENCH) ---
    print("\n[PHASE 1] Loading Training Data (RAGBench Combined)...")
    combined_train_activations = {at: {"not_hallucinated": {}, "hallucinated": {}} for at in act_types}

    for dataset_arg in train_datasets:
        ds_folder = dataset_arg.replace("ragbench_", "")
        dataset_id = dataset_arg
        
        config.BASE_PROJECT_DIR = os.path.join(config.RUNS_DIR, ds_folder)
        config.ACTIVATION_CACHE_DIR = os.path.join(config.BASE_PROJECT_DIR, "activation_cache")
        
        print(f"  📥 Loading {ds_folder}...")
        for label_name in ["not_hallucinated", "hallucinated"]:
            if not check_activations_exist([], config.TARGET_LAYERS, act_types, dataset_id, label_name, check_only_dir=True):
                print(f"❌ Error: Activations for {dataset_id} not found.")
                return

            loaded_group = load_activations_for_group([], config.TARGET_LAYERS, act_types, dataset_id, label_name, load_all=True)
            for at in act_types:
                for k, v in loaded_group[at].items():
                    unique_id = f"{ds_folder}_{k}"
                    combined_train_activations[at][label_name][unique_id] = v

    # --- 2. LOAD TESTING DATA (NQ-SWAP COMBINED) ---
    print(f"\n[PHASE 2] Loading Cross-Test Data (NQ-Swap)...")
    test_activations = {at: {"not_hallucinated": {}, "hallucinated": {}} for at in act_types}
    
    for test_dataset in target_test_datasets:
        mode = "contextual" if "contextual" in test_dataset else "parametric"
        ds_folder = f"nq_swap_project_{mode}"
        dataset_id = test_dataset
        
        config.BASE_PROJECT_DIR = os.path.join(config.RUNS_DIR, ds_folder)
        config.ACTIVATION_CACHE_DIR = os.path.join(config.BASE_PROJECT_DIR, "activation_cache")
        print(f"  📥 Loading {ds_folder} (ID: {dataset_id})...")

        for label_name in ["not_hallucinated", "hallucinated"]:
            if not check_activations_exist([], config.TARGET_LAYERS, act_types, dataset_id, label_name, check_only_dir=True):
                print(f"❌ Error: Activations for {dataset_id} not found.")
                return

            loaded_group = load_activations_for_group([], config.TARGET_LAYERS, act_types, dataset_id, label_name, load_all=True)
            for at in act_types:
                for k, v in loaded_group[at].items():
                    unique_id = f"{dataset_id}_{k}"
                    test_activations[at][label_name][unique_id] = v

    # --- 3. TRAIN AND EVALUATE ---
    print("\n[PHASE 3] Training and Cross-Evaluation...")
    
    if len(target_test_datasets) == 1:
        experiment_folder = f"train_rag_test_{target_test_datasets[0]}"
    else:
        experiment_folder = "train_rag_test_nq_swap_combined"

    config.BASE_PROJECT_DIR = os.path.join(config.RUNS_DIR, experiment_folder)
    base_results_dir = os.path.join(config.BASE_PROJECT_DIR, "results")
    config.IMAGES_DIR = os.path.join(config.BASE_PROJECT_DIR, "images")
    
    probe_types = ["logistic_regression", "ffn", "svm"]
    num_runs = 5

    for p_type in probe_types:
        print(f"\n--- Model Type: {p_type.upper()} ---")
        p_type_dir = os.path.join(base_results_dir, p_type)
        
        for act_type in act_types:
            act_dir = os.path.join(p_type_dir, act_type)
            os.makedirs(act_dir, exist_ok=True)
            
            for layer_idx in config.TARGET_LAYERS:
                X_rag, y_rag = prepare_probing_data(combined_train_activations, layer_idx, act_type)
                X_nq_test, y_nq_test = prepare_probing_data(test_activations, layer_idx, act_type)

                if len(X_rag) < 15 or len(X_nq_test) == 0:
                    continue

                # NEW: Initialize dictionary to collect metrics across the 5 runs
                layer_metrics_history = {"ACC": [], "F1": [], "AUC": [], "AUPRC": []}

                for run_idx in range(num_runs):
                    current_seed = 42 + run_idx
                    
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X_rag, y_rag, test_size=0.30, random_state=current_seed, stratify=y_rag
                    )
                    X_val, _, y_val, _ = train_test_split(
                        X_temp, y_temp, test_size=0.50, random_state=current_seed, stratify=y_temp
                    )

                    probe = train_single_probe(X_train, y_train, config, probe_type=p_type, seed=current_seed)
                    save_probe(probe, layer_idx, act_type, custom_model_name_for_saving, config, run_idx=run_idx, probe_type=p_type)

                    nq_metrics = compute_probe_metrics(probe, X_nq_test, y_nq_test, config.DEVICE)
                    
                    # NEW: Append to history tracker
                    for metric_name, value in nq_metrics.items():
                        layer_metrics_history[metric_name].append(value)
                    
                    with open(os.path.join(act_dir, f"metrics_layer{layer_idx}_run{run_idx}.json"), "w") as f:
                        json.dump(nq_metrics, f)
                
                # NEW: Calculate Mean and Standard Deviation and save summary
                summary_metrics = {}
                for metric_name, values in layer_metrics_history.items():
                    summary_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                    summary_metrics[f"{metric_name}_std"] = float(np.std(values))
                
                summary_filename = f"metrics_layer{layer_idx}_summary.json"
                with open(os.path.join(act_dir, summary_filename), "w") as f:
                    json.dump(summary_metrics, f, indent=4)
                
                print(f"  Layer {layer_idx} {act_type} completed (5 runs + summary).")

    # --- 4. VISUALIZATION ---
    print("\n[PHASE 4] Generating Comparative Plots...")
    original_results_dir = getattr(config, 'RESULTS_DIR', None)
    
    for p_type in probe_types:
        config.RESULTS_DIR = os.path.join(base_results_dir, p_type)
        print(f"  📊 Plotting {p_type.upper()} results...")
        plot_hallucination_metrics(f"{experiment_folder}_{p_type}")
    
    if original_results_dir:
        config.RESULTS_DIR = original_results_dir

    print(f"\n🚀 Reverse experiment finished! Results in: {experiment_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_test_datasets", nargs='+', required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        config.DEVICE = "cpu"

    run_reverse_experiment(config.MODEL_ID, args.target_test_datasets)