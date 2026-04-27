# utils/data_loaders.py
import os
import pandas as pd
from datasets import load_dataset
import config

def load_nq_swap():
    """Carica il dataset NQ-Swap locale."""
    path = os.path.join(config.DATA_DIR, "NQ-Swap.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset non trovato in {path}")
    return pd.read_parquet(path)

def load_ragbench_subset(subset_name: str, split: str = "test"):
    """Carica un subset di RAGBench da Hugging Face."""
    print(f"📥 Caricamento RAGBench ({subset_name})...")
    dataset = load_dataset("galileo-ai/ragbench", subset_name, split=split)
    return dataset.to_pandas()