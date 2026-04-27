import torch
import os

# --- Project Root Detection ---
# This anchors all paths to the location of config.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# --- Model Configuration ---
MODEL_ID = "google/gemma-3-4b-it"
RETRIEVER_MODEL_ID = "google/embeddinggemma-300m"

# Environment Detection
HAS_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if HAS_CUDA else "cpu")

# Scalability Flags
USE_4BIT = False
LOAD_IN_HALF_PRECISION = True

# --- RAG Configuration ---
USE_RAG = False
SKIP_GENERATION = False
RAG_SUBSET = "covidqa"

# Memory Management for 8GB RAM
TOP_K = 2 # Number of documents to retrieve
MAX_SEQ_LENGTH = 1024 # Max tokens length

# --- Project Directory Structure (Absolute Paths) ---
# We use os.path.join(BASE_DIR, ...) to ensure they work from anywhere
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RUNS_DIR = os.path.join(BASE_DIR, "runs")

# Dataset and project naming
DATASET_NAME = RAG_SUBSET if USE_RAG else "nq_swap"
PARAMETRIC_BASELINE = False  # Set to True per fare l'esperimento senza contesto
BASE_PROJECT_DIR = os.path.join(RUNS_DIR, "nq_swap_project" if not USE_RAG else RAG_SUBSET)

RESULTS_DIR = os.path.join(BASE_PROJECT_DIR, "results")
IMAGES_DIR = os.path.join(BASE_PROJECT_DIR, "images")
ACTIVATION_CACHE_DIR = os.path.join(BASE_PROJECT_DIR, "activation_cache")

# --- Experiment Hyperparameters ---
TARGET_LAYERS = list(range(32))
CACHE_FILE_NAME = "nq_swap_cache_results.csv" if not USE_RAG else f"{RAG_SUBSET}_cache_results.csv"

# Ensure directories exist
# for path in [MODELS_DIR, DATA_DIR, RESULTS_DIR, IMAGES_DIR, ACTIVATION_CACHE_DIR]:
    # os.makedirs(path, exist_ok=True)