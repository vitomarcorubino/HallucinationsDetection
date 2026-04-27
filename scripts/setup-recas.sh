#!/bin/bash

# Path to the project directory on ReCaS
cd /lustrehome/vitomarcorubino/HallucinationsDetection

# Create and activate a virtual environment using uv
# uv venv --python 3.12
python3 -m venv .venv
source .venv/bin/activate

# rm -rf ~/.cache/pip
# rm -rf ~/.cache/uv
# rm -rf ~/.cache/python
# 
# uv pip sync requirements.lock

# Upgrade pip to the latest version and install required packages
pip install --upgrade pip
pip install -r requirements.txt

# Download used models from Hugging Face
huggingface-cli download google/gemma-3-4b-it
huggingface-cli download google/embeddinggemma-300m

# huggingface-cli download google/gemma-2-9b-it
# huggingface-cli download google/gemma-scope-9b-pt-res --include "layer_**/width_16k/**"

# huggingface-cli download meta-llama/Llama-3.1-8B-Instruct
# huggingface-cli download OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x
