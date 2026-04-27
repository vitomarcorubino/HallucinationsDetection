#!/bin/bash
# 1. Vai nella cartella del progetto
cd /lustrehome/vitomarcorubino/HallucinationsDetection

# 2. Carica l'ambiente virtuale
source .venv/bin/activate

# 3. Esegui Python passando tutti gli argomenti che arrivano da HTCondor
# "$@" serve a passare esattamente la stringa definita in 'arguments' del file .htc
python -W ignore "$@"