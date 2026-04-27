import os
import torch
import torch.optim as optim
import numpy as np
import joblib 
from torch.utils.data import DataLoader
from sklearn.svm import LinearSVC
from .model import LogisticRegressionProbe, FFNProbe
from .data_prep import ActivationDataset

def train_single_probe(X, y, config, probe_type="logistic_regression", epochs=50, l1_lambda=0.0006, seed=42):
    """
    Trains a probe (LR, FFN, or SVM) using a specific seed for reproducibility 
    and variability across multiple runs.
    """
    # Impostiamo il seed per garantire che ogni 'run' sia diversa ma riproducibile
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Logica per Modelli PyTorch (LR e FFN) ---
    if probe_type in ["logistic_regression", "ffn"]:
        dataset = ActivationDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=min(len(X), 32), shuffle=True)

        # Bilanciamento classi (fondamentale per le allucinazioni)
        num_pos = (y == 1).sum().item() if torch.is_tensor(y) else (y == 1).sum()
        num_neg = (y == 0).sum().item() if torch.is_tensor(y) else (y == 0).sum()
        pos_weight = torch.tensor([num_neg / num_pos if num_pos > 0 else 1.0]).to(config.DEVICE)

        input_dim = X.shape[1]
        if probe_type == "logistic_regression":
            probe = LogisticRegressionProbe(input_dim).to(config.DEVICE)
        else:
            probe = FFNProbe(input_dim).to(config.DEVICE)

        optimizer = optim.Adam(probe.parameters(), lr=0.002)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        probe.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                batch_X, batch_y = batch_X.to(config.DEVICE), batch_y.to(config.DEVICE)
                optimizer.zero_grad()
                outputs = probe(batch_X)
                loss = criterion(outputs, batch_y.view_as(outputs))
                
                # Regolarizzazione L1 per Logistic Regression (Sparse Probing)
                if probe_type == "logistic_regression": 
                    l1_penalty = sum(p.abs().sum() for p in probe.parameters())
                    loss += l1_lambda * l1_penalty

                loss.backward()
                optimizer.step()
        return probe

    # --- Logica per SVM (Scikit-Learn) ---
    elif probe_type == "svm":
        X_np = X.cpu().numpy() if torch.is_tensor(X) else X
        y_np = y.cpu().numpy() if torch.is_tensor(y) else y
        
        # Usiamo il seed dinamico per random_state
        model = LinearSVC(
            class_weight='balanced', 
            max_iter=2000, 
            dual=False, 
            random_state=seed 
        )
        model.fit(X_np, y_np)
        return model

def save_probe(probe, layer_idx, act_type, model_name, config, run_idx=0, probe_type="logistic_regression"):
    """
    Saves the probe in a dedicated subfolder based on the prober type.
    """
    model_folder_name = model_name.split('/')[-1]
    # Creiamo una sottocartella per tipo di prober (es: models/gemma-3-4b-it/svm/)
    target_dir = os.path.join(config.MODELS_DIR, model_folder_name, probe_type)
    os.makedirs(target_dir, exist_ok=True)

    filename = f"probe_L{layer_idx}_{act_type}_run{run_idx}"
    
    if isinstance(probe, torch.nn.Module):
        save_path = os.path.join(target_dir, f"{filename}.pt")
        torch.save(probe.state_dict(), save_path)
    else:
        # Salvataggio SVM tramite joblib
        save_path = os.path.join(target_dir, f"{filename}.joblib")
        joblib.dump(probe, save_path)

    print(f"✅ {probe_type.upper()} saved: {os.path.basename(save_path)}")