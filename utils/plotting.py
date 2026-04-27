import os
import re
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages # Native PDF multi-page support
import config
import numpy as np

def load_all_metrics(results_dir):
    """
    Recursively scans the results directory to build a DataFrame,
    now supporting multiple runs for confidence intervals.
    """
    all_records = []

    for root, dirs, files in os.walk(results_dir):
        for file in files:
            if file.startswith("metrics_layer") and file.endswith(".json"):
                path = os.path.join(root, file)

                # Extract layer, activation type and run_id
                layer_match = re.search(r'layer(\d+)', file)
                run_match = re.search(r'run(\d+)', file)
                act_type = os.path.basename(root)

                if layer_match:
                    with open(path, "r") as f:
                        m = json.load(f)
                        m["layer"] = int(layer_match.group(1))
                        m["activation"] = act_type
                        m["run_id"] = int(run_match.group(1)) if run_match else 0
                        all_records.append(m)

    return pd.DataFrame(all_records)


def plot_hallucination_metrics(dataset_name):
    """
    Generates PNG plots with confidence intervals and compiles them into a single PDF.
    """
    df = load_all_metrics(config.RESULTS_DIR)
    if df.empty:
        print("❌ No metrics found to plot.")
        return

    model_safe_name = config.MODEL_ID.split("/")[-1]
    save_dir = os.path.join(config.IMAGES_DIR, "hallucination_detection")
    os.makedirs(save_dir, exist_ok=True)

    metrics_to_plot = ["ACC", "AUC", "AUPRC", "F1"]
    palette = {"hidden": "red", "mlp": "blue", "attn": "green"}
    
    # We will collect figures to save them into a single PDF later
    pdf_path = os.path.join(save_dir, f"{model_safe_name}_{dataset_name}_FULL_REPORT.pdf")
    
    with PdfPages(pdf_path) as pdf:
        for metric in metrics_to_plot:
            if metric not in df.columns or df[metric].isna().all():
                continue

            # Plotting with Confidence Intervals (Seaborn does this automatically if multiple run_ids exist)
            plt.figure(figsize=(10, 6), dpi=150)
            sns.lineplot(
                data=df, 
                x="layer", 
                y=metric, 
                hue="activation", 
                palette=palette, 
                marker='o',
                errorbar=('ci', 95) # Adds the "shadow" (95% confidence interval)
            )

            plt.title(f"{metric} across layers - {model_safe_name}\n({dataset_name})")
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0, 1.05)
            plt.legend(title="Activation", loc='best')

            # 1. Save individual PNG
            png_filename = f"{model_safe_name}_{dataset_name}_{metric}.png"
            plt.savefig(os.path.join(save_dir, png_filename), format='png', bbox_inches='tight')
            
            # 2. Add the current figure to the multi-page PDF
            pdf.savefig()
            plt.close()

    print(f"✅ Individual PNGs and compiled PDF saved to {save_dir}")


def plot_activation_distribution(layer_data_types, layer_idx, save_dir, pdf_pages=None):
    """
    Genera una figura con 3 subplot (Hidden, Attn, MLP).
    Ogni subplot confronta Faithful vs Hallucinated per quel tipo.
    Salva un PNG individuale e aggiunge la pagina al PDF se fornito.
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=150)
    color_map = {"not_hallucinated": "blue", "hallucinated": "red"}
    act_types = ["hidden", "attn", "mlp"]
    
    for i, act_type in enumerate(act_types):
        ax = axes[i]
        if act_type not in layer_data_types:
            ax.set_title(f"{act_type.upper()} (No Data)")
            continue
            
        for label, color in color_map.items():
            X = layer_data_types[act_type].get(label)
            if X is None or X.numel() == 0: continue
            
            # Campionamento per prestazioni
            sample_values = X.flatten().cpu().numpy()
            if len(sample_values) > 30000:
                sample_values = np.random.choice(sample_values, 30000, replace=False)
            
            sns.kdeplot(sample_values, fill=True, color=color, label=label, alpha=0.3, ax=ax)
        
        ax.set_title(f"{act_type.upper()} Distribution")
        ax.set_xlabel("Value")
        ax.legend()
        ax.grid(True, alpha=0.2)

    plt.suptitle(f"KDE Spectral Analysis - Layer {layer_idx}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Salva PNG individuale
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"KDE_Layer_{layer_idx}_Subplots.png"))
    
    # Aggiungi al PDF se il contesto è aperto
    if pdf_pages:
        pdf_pages.savefig(fig)
        
    plt.close(fig)

def plot_structural_analysis_report(stats_df, dataset_name, save_dir):
    """
    Genera un PDF con 3 grafici affiancati per metrica:
    1. Overlaid (Faithful vs Hallucinated per tutti i tipi)
    2. Solo Faithful
    3. Solo Hallucinated
    """
    pdf_path = os.path.join(save_dir, f"Structural_Analysis_{dataset_name}.pdf")
    metrics = ["Kurtosis", "Hoyer", "Gini", "L1", "L2"]
    color_map = {"hidden": "red", "attn": "green", "mlp": "blue"}
    
    with PdfPages(pdf_path) as pdf:
        for metric in metrics:
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            
            # --- Grafico 1: Overlaid (6 linee) ---
            ax0 = axes[0]
            for act_type in ["hidden", "attn", "mlp"]:
                # Faithful (Linea continua)
                sf = stats_df[(stats_df['activation'] == act_type) & (stats_df['group'] == "not_hallucinated")]
                ax0.plot(sf['layer'], sf[metric], label=f"{act_type} (F)", 
                         color=color_map[act_type], linestyle='-', marker='o', alpha=0.7)
                # Hallucinated (Linea tratteggiata)
                sh = stats_df[(stats_df['activation'] == act_type) & (stats_df['group'] == "hallucinated")]
                ax0.plot(sh['layer'], sh[metric], label=f"{act_type} (H)", 
                         color=color_map[act_type], linestyle='--', marker='x', alpha=0.7)
            ax0.set_title(f"{metric} - Overlaid Comparison")
            ax0.legend(fontsize='small', ncol=2)
            ax0.grid(True, alpha=0.3)

            # --- Grafico 2: Solo Faithful ---
            ax1 = axes[1]
            for act_type in ["hidden", "attn", "mlp"]:
                s = stats_df[(stats_df['activation'] == act_type) & (stats_df['group'] == "not_hallucinated")]
                ax1.plot(s['layer'], s[metric], label=act_type, color=color_map[act_type], marker='o')
            ax1.set_title(f"{metric} - Faithful Only")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # --- Grafico 3: Solo Hallucinated ---
            ax2 = axes[2]
            for act_type in ["hidden", "attn", "mlp"]:
                s = stats_df[(stats_df['activation'] == act_type) & (stats_df['group'] == "hallucinated")]
                ax2.plot(s['layer'], s[metric], label=act_type, color=color_map[act_type], marker='x')
            ax2.set_title(f"{metric} - Hallucinated Only")
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            plt.suptitle(f"{metric} Analysis: {dataset_name}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)