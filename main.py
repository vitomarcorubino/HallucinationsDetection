import os
import argparse
import pandas as pd
import gc
import json # <-- Aggiunto per il salvataggio
import config
import numpy as np
# Module Imports
from inference.model_handler import authenticate_huggingface, load_transformer_model
from inference.predictor import ModelPredictor
from inference.retriever import GemmaRetriever
from activations.extractor import ActivationExtractor
from activations.layer_configs import get_layer_map
from activations.cache_utils import check_activations_exist, load_activations_for_group
from probing.trainer import train_single_probe, save_probe
from probing.data_prep import prepare_probing_data
from utils.plotting import plot_hallucination_metrics
from utils.data_loaders import load_nq_swap, load_ragbench_subset
from sklearn.model_selection import train_test_split
from utils.evaluation import compute_probe_metrics


def run_experiment():
    # --- 0. Dynamic Path Update ---
    if config.USE_RAG:
        dataset_folder = config.RAG_SUBSET
    else:
        suffix = "parametric" if getattr(config, 'PARAMETRIC_BASELINE', False) else "contextual"
        dataset_folder = f"nq_swap_project_{suffix}"

    config.BASE_PROJECT_DIR = os.path.join(config.RUNS_DIR, dataset_folder)
    config.RESULTS_DIR = os.path.join(config.BASE_PROJECT_DIR, "results")
    config.IMAGES_DIR = os.path.join(config.BASE_PROJECT_DIR, "images")
    config.ACTIVATION_CACHE_DIR = os.path.join(config.BASE_PROJECT_DIR, "activation_cache")

    for path in [config.RESULTS_DIR, config.IMAGES_DIR, config.ACTIVATION_CACHE_DIR]:
        os.makedirs(path, exist_ok = True)

    # --- 1. Setup Phase ---
    authenticate_huggingface()
    model, tokenizer = load_transformer_model(config.MODEL_ID)

    extractor = ActivationExtractor(model, tokenizer, config)
    layer_map = get_layer_map(config.MODEL_ID, config.TARGET_LAYERS)
    act_types = list(layer_map.keys())

    # --- 2. Phase: Data Loading and Inference ---
    cache_path = os.path.join(config.BASE_PROJECT_DIR, config.CACHE_FILE_NAME)

    if os.path.exists(cache_path):
        print(f"✅ [CACHE] Loading existing predictions from {cache_path}")
        labeled_df = pd.read_csv(cache_path)

        if not config.USE_RAG:
            print("🔗 Re-linking NQ-Swap original data...")
            full_df = load_nq_swap()
            for col in ['sub_context', 'question']:
                if col not in labeled_df.columns:
                    labeled_df[col] = labeled_df.index.map(lambda idx: full_df.iloc[idx][col])
    else:
        if config.USE_RAG:
            print(f"🌍 Loading RAGBench subset: {config.RAG_SUBSET}")
            full_df = load_ragbench_subset(config.RAG_SUBSET)
            test_df = full_df.copy()

            if config.SKIP_GENERATION:
                print(f"✅ Option B: Using pre-existing responses.")
                test_df['model_prediction'] = test_df['response']
                test_df['label'] = 1 - test_df['adherence_score']
                test_df['sub_context'] = test_df['documents'].apply(lambda x: "\n".join(x))
                labeled_df = test_df
                labeled_df.to_csv(cache_path, index = False)
            else:
                print("🔍 Running RAG Retrieval + Generation...")
                retriever = GemmaRetriever(config.RETRIEVER_MODEL_ID)
                retrieved_contexts = [retriever.get_top_k(r['question'], r['documents']) for _, r in test_df.iterrows()]
                test_df['sub_context'] = retrieved_contexts
                del retriever
                gc.collect()

                predictor = ModelPredictor(model, tokenizer, config)
                prompt_template = (
                    "Read the following context and answer the question.\n\n"
                    "Context: {context}\n\n"
                    "Question: {question}\n\n"
                    "Answer:"
                )
                labeled_df = predictor.run_inference(test_df, prompt_template, context_col = "sub_context",
                                                     output_cache_path = cache_path)
        else:
            print("🚀 Starting NQ-Swap Experiment...")
            full_df = load_nq_swap()
            test_df = full_df.copy()

            predictor = ModelPredictor(model, tokenizer, config)

            if getattr(config, 'PARAMETRIC_BASELINE', False):
                print("⚠️ Mode: Parametric Baseline (No Context)")
                prompt_template = "Answer the following question with a single word or number. Do not write full sentences.\n\nQuestion: {question}\n\nAnswer:"
            else:
                print("📝 Mode: Contextual Conflict (With Swapped Context)")
                prompt_template = (
                    "Read the following context and answer the question with a single word or number.\n\n"
                    "Context: {context}\n\n"
                    "Question: {question}\n\n"
                    "Answer:"
                )

            labeled_df = predictor.run_inference(test_df, prompt_template, output_cache_path = cache_path)

    # --- TASK 3: Calculate and Print Hallucination Rate ---
    total_samples = len(labeled_df)
    count_faithful = (labeled_df['label'] == 0).sum()
    count_hallucination = (labeled_df['label'] == 1).sum()
    count_noise = (labeled_df['label'] == -1).sum()

    f_rate = (count_faithful / total_samples) * 100 if total_samples > 0 else 0
    h_rate = (count_hallucination / total_samples) * 100 if total_samples > 0 else 0
    n_rate = (count_noise / total_samples) * 100 if total_samples > 0 else 0

    print("\n" + "=" * 50)
    print(f"📊 SYSTEM PERFORMANCE REPORT")
    print(f"Dataset: {config.DATASET_NAME.upper()}")
    print(f"Total Samples: {total_samples}")
    print("-" * 30)
    print(f"✅ Faithful (Label 0):      {count_faithful} ({f_rate:.2f}%)")
    print(f"🧠 Parametric Bias (Label 1): {count_hallucination} ({h_rate:.2f}%)")
    print(f"📉 Noise/Other (Label -1):   {count_noise} ({n_rate:.2f}%)")
    print("=" * 50 + "\n")

    labeled_df = labeled_df[labeled_df['label'] != -1].reset_index(drop = True)
    if count_noise > 0:
        print(f"🧹 Removed {count_noise} noisy samples. Proceeding to Probing with {len(labeled_df)} clean samples.")

    # --- 3. Phase: Activation Extraction ---
    dataset_id = config.DATASET_NAME
    activations_storage = {at: {"not_hallucinated": {}, "hallucinated": {}} for at in act_types}

    for label_val, label_name in [(0, "not_hallucinated"), (1, "hallucinated")]:
        sub_df = labeled_df[labeled_df['label'] == label_val]
        if sub_df.empty: continue

        sub_ids = sub_df.index.tolist()

        if check_activations_exist(sub_ids, config.TARGET_LAYERS, act_types, dataset_id, label_name):
            print(f"✅ [CACHE] '{label_name}' activations found.")
        else:
            print(f"\n--- Extracting {label_name} Activations ---")
            samples_to_extract = []
            for idx, row in sub_df.iterrows():
                if getattr(config, 'PARAMETRIC_BASELINE', False) and not config.USE_RAG:
                    full_text = f"Answer the following question.\n\nQuestion: {row['question']}\n\nAnswer: {row['model_prediction']}"
                else:
                    context_val = row['sub_context'] if 'sub_context' in row else row.get('context', '')
                    full_text = (
                        f"Read the following context and answer the question.\n\n"
                        f"Context: {context_val}\n\n"
                        f"Question: {row['question']}\n\n"
                        f"Answer: {row['model_prediction']}"
                    )
                samples_to_extract.append((full_text, idx))
            extractor.extract_from_dataset(samples_to_extract, dataset_id, label_name, layer_map)

        loaded_group = load_activations_for_group(sub_ids, config.TARGET_LAYERS, act_types, dataset_id, label_name)
        for at in act_types:
            activations_storage[at][label_name] = loaded_group[at]

    # --- 4. Phase: Probing ---
    print("\n--- Starting Probing Phase (N Runs x Probe Types) ---")
    num_runs = 5
    probe_types = ["logistic_regression", "ffn", "svm"]

    for p_type in probe_types:
        print(f"\n[{p_type.upper()}] Training and Evaluation...")
        
        probe_results_dir = os.path.join(config.RESULTS_DIR, p_type)
        os.makedirs(probe_results_dir, exist_ok=True)

        for act_type in act_types:
            act_dir = os.path.join(probe_results_dir, act_type)
            os.makedirs(act_dir, exist_ok=True)
            
            for layer_idx in config.TARGET_LAYERS:
                X, y = prepare_probing_data(activations_storage, layer_idx, act_type)

                if len(X) < 15:
                    if p_type == probe_types[0]: 
                        print(f"Skipping layer {layer_idx} {act_type}: not enough data for 3-way split.")
                    continue

                # NEW: Initialize a dictionary to collect metrics across the 5 runs
                layer_metrics_history = {"ACC": [], "F1": [], "AUC": [], "AUPRC": []}

                for run_idx in range(num_runs):
                    current_seed = 42 + run_idx
                    
                    X_train, X_temp, y_train, y_temp = train_test_split(
                        X, y, test_size=0.30, random_state=current_seed, stratify=y
                    )
                    X_val, X_test, y_val, y_test = train_test_split(
                        X_temp, y_temp, test_size=0.50, random_state=current_seed, stratify=y_temp
                    )

                    probe = train_single_probe(X_train, y_train, config, probe_type=p_type, seed=current_seed)
                    save_probe(probe, layer_idx, act_type, config.MODEL_ID, config, run_idx=run_idx, probe_type=p_type)

                    test_metrics = compute_probe_metrics(probe, X_test, y_test, config.DEVICE)
                    
                    # NEW: Append the results of this specific run to our history tracker
                    for metric_name, value in test_metrics.items():
                        layer_metrics_history[metric_name].append(value)
                    
                    if run_idx == 0: 
                        print(f"L{layer_idx} {act_type} ({p_type}) | Test ACC: {test_metrics['ACC']:.3f} | Test AUC: {test_metrics['AUC']:.3f} (Run {run_idx})")

                    json_filename = f"metrics_layer{layer_idx}_run{run_idx}.json"
                    with open(os.path.join(act_dir, json_filename), "w") as f:
                        json.dump(test_metrics, f)

                # NEW: Once all 5 runs are complete, calculate Mean and Standard Deviation
                summary_metrics = {}
                for metric_name, values in layer_metrics_history.items():
                    summary_metrics[f"{metric_name}_mean"] = float(np.mean(values))
                    summary_metrics[f"{metric_name}_std"] = float(np.std(values))
                
                # NEW: Save the aggregated summary JSON for this specific layer
                summary_filename = f"metrics_layer{layer_idx}_summary.json"
                with open(os.path.join(act_dir, summary_filename), "w") as f:
                    json.dump(summary_metrics, f, indent=4)

    # --- 5. Phase: Visualization ---
    original_results_dir = config.RESULTS_DIR
    
    for p_type in probe_types:
        # Inganniamo config per far puntare il plotter alla sottocartella corretta
        config.RESULTS_DIR = os.path.join(original_results_dir, p_type)
        print(f"\n📊 Generating plots with confidence intervals for {p_type.upper()}...")
        # Nome univoco per il plot, es: ragbench_covidqa_svm
        plot_hallucination_metrics(f"{dataset_id}_{p_type}")
        
    # Ripristiniamo config per sicurezza
    config.RESULTS_DIR = original_results_dir

    print(f"\n🚀 Pipeline finished for {dataset_id}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Hallucination Detection Experiment")

    parser.add_argument("--model_id", type = str, default = config.MODEL_ID)
    parser.add_argument("--dataset", type = str, default = "nq_swap")
    parser.add_argument("--subset", type = str, default = "factual")
    parser.add_argument("--use_rag", action = "store_true")
    parser.add_argument("--skip_generation", action = "store_true")
    parser.add_argument("--top_k", type = int, default = config.TOP_K)
    parser.add_argument("--max_seq_length", type = int, default = config.MAX_SEQ_LENGTH)

    parser.add_argument("--parametric_baseline",
                        action = "store_true",
                        default = config.PARAMETRIC_BASELINE, 
                        help = "Run NQ-Swap without context")

    args = parser.parse_args()

    config.MODEL_ID = args.model_id
    config.USE_RAG = args.use_rag
    config.SKIP_GENERATION = args.skip_generation
    config.TOP_K = args.top_k
    config.MAX_SEQ_LENGTH = args.max_seq_length
    config.PARAMETRIC_BASELINE = args.parametric_baseline

    if args.use_rag:
        config.RAG_SUBSET = args.subset
        config.DATASET_NAME = f"ragbench_{args.subset}"
        config.CACHE_FILE_NAME = f"{args.subset}_cache_results.csv"
    else:
        mode_suffix = "parametric" if config.PARAMETRIC_BASELINE else "contextual"
        config.DATASET_NAME = f"nq_swap_{mode_suffix}"
        config.CACHE_FILE_NAME = f"nq_swap_{mode_suffix}_cache_results.csv"

    print(f"🛠️ Experiment Mode: {config.DATASET_NAME.upper()}")
    run_experiment()