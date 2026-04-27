import os
import pandas as pd
import argparse
import torch
from matplotlib.backends.backend_pdf import PdfPages
import config
from activations.layer_configs import get_layer_map
from utils.scoring import compute_activation_stats
from utils.plotting import plot_activation_distribution, plot_structural_analysis_report

def run_activation_analysis(dataset_id):
    print(f"🚀 Starting Structural Analysis for: {dataset_id}")
    
    # 1. Setup percorsi
    if "ragbench" in dataset_id:
        ds_folder = dataset_id.replace("ragbench_", "")
        project_dir = os.path.join(config.RUNS_DIR, ds_folder)
        csv_name = f"{ds_folder}_cache_results.csv"
    else:
        suffix = dataset_id.split('_')[-1]
        ds_folder = f"nq_swap_project_{suffix}"
        project_dir = os.path.join(config.RUNS_DIR, ds_folder)
        csv_name = f"{dataset_id}_cache_results.csv"

    analysis_dir = os.path.join(project_dir, "analysis")
    kde_dir = os.path.join(analysis_dir, "kde_plots")
    os.makedirs(kde_dir, exist_ok=True)

    csv_path = os.path.join(project_dir, csv_name)
    if not os.path.exists(csv_path):
        print(f"❌ Error: Cache CSV not found at {csv_path}")
        return

    df_results = pd.read_csv(csv_path)
    group_ids = {
        "not_hallucinated": df_results[df_results['label'] == 0].index.tolist(),
        "hallucinated": df_results[df_results['label'] == 1].index.tolist()
    }
    
    layer_map = get_layer_map(config.MODEL_ID, config.TARGET_LAYERS)
    act_types = list(layer_map.keys())
    all_stats = []
    model_folder = config.MODEL_ID.split('/')[-1]

    # Prepara il PDF unico per i KDE plots
    kde_pdf_path = os.path.join(analysis_dir, f"KDE_Full_Report_{dataset_id}.pdf")
    
    with PdfPages(kde_pdf_path) as kde_pdf:
        # 2. Loop di analisi su TUTTI i Layer
        for layer_idx in config.TARGET_LAYERS:
            print(f"🔄 Processing Layer {layer_idx}...")
            kde_layer_collector = {}

            for act_type in act_types:
                layer_data_by_label = {}
                for label, ids in group_ids.items():
                    if not ids: continue
                    
                    cache_dir = os.path.join(
                        project_dir, "activation_cache", model_folder, 
                        dataset_id, f"activation_{act_type}", label
                    )
                    
                    tensors = []
                    for sample_id in ids:
                        file_path = os.path.join(cache_dir, f"layer{layer_idx}-id{sample_id}.pt")
                        if os.path.exists(file_path):
                            try:
                                t = torch.load(file_path, map_location="cpu", weights_only=True)
                                t = t.flatten()
                                if t.shape[0] > 0: tensors.append(t)
                            except: continue
                    
                    if tensors:
                        X_tensor = torch.stack(tensors, dim=0)
                        stats = compute_activation_stats(X_tensor)
                        stats.update({"layer": layer_idx, "activation": act_type, "group": label})
                        all_stats.append(stats)
                        layer_data_by_label[label] = X_tensor
                
                kde_layer_collector[act_type] = layer_data_by_label

            # 3. Generazione KDE Plot per OGNI layer (PNG + PDF)
            plot_activation_distribution(kde_layer_collector, layer_idx, kde_dir, pdf_pages=kde_pdf)

    # 4. Generazione Report Strutturale PDF (Layout 3 grafici)
    if all_stats:
        print("📊 Generating final structural report...")
        df_stats = pd.DataFrame(all_stats)
        plot_structural_analysis_report(df_stats, dataset_id, analysis_dir)
        print(f"✅ Analysis completed! Results in: {analysis_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    args = parser.parse_args()
    run_activation_analysis(args.dataset)