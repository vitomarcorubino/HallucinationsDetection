import os
import yaml
import itertools

# --- Configurazione Percorsi Dinamica ---
# Trova la cartella dove si trova fisicamente questo script (HallucinationsDetection/scripts)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Torna su di un livello per trovare la radice del progetto (HallucinationsDetection)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Se sei su ReCaS, vogliamo il percorso Linux, se sei su Windows quello locale.
# Questa logica di PROJECT_ROOT ora è corretta per entrambi.
config_path = os.path.join(PROJECT_ROOT, "scripts", "config_experiment.yaml")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "generated_jobs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)

# Carica la configurazione YAML
with open(config_path) as f:
    cfg = yaml.safe_load(f)

common_keys = list(cfg["common"].keys())
common_values = list(cfg["common"].values())

submit_lines = []

# Iteriamo sugli scenari (RAG vs No-RAG) definiti nello YAML
for scenario in cfg["scenarios"]:
    use_rag = scenario["use_rag"]

    for data_name, subsets in scenario["datasets"].items():
        for sub in subsets:
            # Genera tutte le combinazioni dei parametri 'common'
            for combo in itertools.product(*common_values):
                params = dict(zip(common_keys, combo))

                # Costruzione argomenti per il tuo main.py
                args = [
                    "main.py",
                    f"--model_id {params['model_id']}",
                    f"--dataset {data_name}",
                    f"--subset {sub}",
                    f"--top_k {params['top_k']}",
                    f"--max_seq_length {params['max_seq_length']}"
                ]

                # Gestione dei flag booleani
                if use_rag:
                    args.append("--use_rag")
                if params.get('skip_generation'):
                    args.append("--skip_generation")

                # Ogni riga deve terminare con 'queue' per HTCondor
                submit_lines.append(f'arguments = "{" ".join(args)}"\nqueue\n')

# --- Scrittura del file .htc ---
submit_filename = os.path.join(OUTPUT_DIR, "run_hallucination_tests.htc")
with open(submit_filename, "w") as f:
    # Header del file HTCondor con percorsi dinamici
    f.write(f"""universe       = vanilla
executable     = {os.path.join(PROJECT_ROOT, ".venv/bin/python")}
request_cpus   = 1
request_gpus   = 1
request_memory = 32GB

initialdir     = {PROJECT_ROOT}
getenv         = True

log    = logs/job_$(Cluster).log
output = logs/job_$(Cluster)_$(Process).out
error  = logs/job_$(Cluster)_$(Process).err

""")
    f.writelines(submit_lines)

print(f"Creato: {submit_filename} con {len(submit_lines)} job.")