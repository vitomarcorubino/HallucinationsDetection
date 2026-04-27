from .logging import save_layer_metrics
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, f1_score

def compute_probe_metrics(probe, X_test, y_test, device):
    """
    Calcola le metriche supportando sia prober PyTorch (LogReg, FFN) 
    che prober Scikit-learn (SVM).
    """
    
    # --- CASO 1: Prober PyTorch (Logistic Regression o FFN) ---
    if hasattr(probe, 'eval'):
        probe.eval()
        with torch.no_grad():
            # Assicuriamoci che X_test sia un tensore
            if isinstance(X_test, np.ndarray):
                test_tensor = torch.from_numpy(X_test).float().to(device)
            else:
                test_tensor = X_test.to(device)

            logits = probe(test_tensor)
            # Se l'output è un singolo valore (batch size 1), lo aggiustiamo
            if logits.dim() == 0:
                logits = logits.unsqueeze(0)
                
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
    # --- CASO 2: Prober Scikit-learn (SVM / LinearSVC) ---
    else:
        # L'SVM vuole NumPy array, non tensori
        X_np = X_test.cpu().numpy() if isinstance(X_test, torch.Tensor) else X_test
        
        # Per l'accuratezza e F1 usiamo predict
        preds = probe.predict(X_np)
        
        # Per l'AUC serve un punteggio continuo. 
        # LinearSVC non ha predict_proba, usiamo decision_function.
        if hasattr(probe, "decision_function"):
            probs = probe.decision_function(X_np)
        else:
            probs = preds # Fallback se mancano i punteggi

    # --- CALCOLO METRICHE (comune a entrambi) ---
    # Assicuriamoci che y_test sia in formato numpy
    y_true = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test

    acc = accuracy_score(y_true, preds)
    f1 = f1_score(y_true, preds, zero_division=0)

    try:
        auc = roc_auc_score(y_true, probs)
        auprc = average_precision_score(y_true, probs)
    except ValueError:
        # Capita se nel test set c'è solo una classe (es. tutte allucinazioni)
        auc, auprc = 0.5, 0.0

    return {"ACC": acc, "F1": f1, "AUC": auc, "AUPRC": auprc}


def run_evaluation_cycle(probe, X, y, layer_idx, act_type, dataset_name, config):
    metrics = compute_probe_metrics(probe, X, y, config.DEVICE)
    save_layer_metrics(dataset_name, act_type, layer_idx, metrics)
    return metrics