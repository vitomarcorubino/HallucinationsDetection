import os
import pandas as pd
import torch
from tqdm import tqdm
# Importiamo la tua funzione di normalizzazione (stile SQuAD)
from utils.text_helpers import normalize_answer


class ModelPredictor:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def run_inference(self, df: pd.DataFrame, prompt_template: str,
                      context_col: str = "sub_context", question_col: str = "question",
                      output_cache_path: str = None):

        # 1. Gestione Cache (Anti-corruzione: evita di caricare predizioni vuote)
        if output_cache_path and os.path.exists(output_cache_path):
            print(f"✅ Loading results from existing cache: {output_cache_path}")
            cached_df = pd.read_csv(output_cache_path)
            if len(cached_df) == len(df) and not cached_df['model_prediction'].isnull().all():
                df['model_prediction'] = cached_df['model_prediction'].fillna("").values
                df['label'] = cached_df['label'].values
                return df
            else:
                print("⚠️ Cache corrupted or empty. Forcing re-generation...")

        # 2. Ciclo di Generazione
        print(f"🚀 Starting inference on {len(df)} samples...")
        predictions = []
        self.model.eval()

        for idx, row in tqdm(df.iterrows(), total = len(df), desc = "Inference"):
            # Gestione sicura per parametric mode dove il contesto manca
            ctx = row[context_col] if context_col in row and isinstance(row[context_col], str) else ""

            prompt = prompt_template.format(
                context = ctx,
                question = row[question_col]
            )
            inputs = self.tokenizer(prompt, return_tensors = "pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens = 30,  # Aumentato leggermente per accomodare le frasi discorsive
                    do_sample = False,
                    pad_token_id = self.tokenizer.eos_token_id
                )
                gen_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:],
                                                 skip_special_tokens = True).strip()
                predictions.append(gen_text)

        df['model_prediction'] = predictions

        # 3. Smart Labeling Logic (Ottimizzato per Verbosità e Normalizzazione SQuAD)
        print("🧠 Labeling predictions (Scientific Logic)...")
        is_parametric_mode = getattr(self.config, 'PARAMETRIC_BASELINE', False)

        def assign_label(row):
            pred = normalize_answer(str(row['model_prediction']))

            # Se la risposta è totalmente vuota, è un errore tecnico (rumore)
            if not pred or pred == "nan":
                return -1

            # Normalizziamo le risposte attese nel dataset
            contextual_gt = normalize_answer(str(row['sub_answer'])) if 'sub_answer' in row else None
            parametric_gt = normalize_answer(str(row['org_answer'])) if 'org_answer' in row else None

            # --- CASO A: Esperimento PARAMETRICO (Solo Memoria) ---
            if is_parametric_mode:
                if parametric_gt and (parametric_gt in pred or pred in parametric_gt):
                    return 0
                return 1

            # --- CASO B: Esperimento CONTEXTUAL (Conflitto Context vs Memory) ---
            elif 'org_answer' in row:
                # 0 = Fedele al Contesto falso (Swapped)
                if contextual_gt and (contextual_gt in pred or pred in contextual_gt):
                    return 0

                # 1 = Cede alla Memoria VERA (Parametric Bias)
                if parametric_gt and (parametric_gt in pred or pred in parametric_gt):
                    return 1

                return -1

            # --- CASO C: Standard RAGBench ---
            else:
                # Se è RAGBench normale, tutto ciò che non è aderente al contesto è 1
                if contextual_gt and (contextual_gt in pred or pred == contextual_gt):
                    return 0
                return 1

        df['label'] = df.apply(assign_label, axis = 1)

        # 4. Salvataggio Cache
        if output_cache_path:
            os.makedirs(os.path.dirname(output_cache_path), exist_ok = True)
            df[['model_prediction', 'label']].to_csv(output_cache_path, index = False)
            print(f"✅ Results cached at {output_cache_path}")

        return df