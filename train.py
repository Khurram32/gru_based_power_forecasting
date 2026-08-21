"""
Train the GRU medium-term electricity-consumption forecaster.

Trains with 1-step-ahead supervision on sliding windows (standard for
recurrent load-forecasting models, consistent with the paper's Ref. [26]
model which is trained on a rolling monthly window of lagged features),
with early stopping on validation MAPE, gradient clipping, and Adam.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from preprocessing.feature_engineering import prepare_data
from models.gru_model import GRUForecaster
from evaluation.metrics import mape as mape_fn


def make_loaders(data: dict):
    def to_loader(X, y, shuffle):
        X_t = torch.from_numpy(X)
        y_t = torch.from_numpy(y)
        ds = TensorDataset(X_t, y_t)
        return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=shuffle)

    train_loader = to_loader(data["X_train"], data["y_train"], shuffle=True)
    val_loader = to_loader(data["X_val"], data["y_val"], shuffle=False)
    return train_loader, val_loader


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(mode=train)
    total_loss = 0.0
    n_samples = 0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            preds = model(X_batch)
            loss = criterion(preds, y_batch)

            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        n_samples += X_batch.size(0)

    return total_loss / max(n_samples, 1)


def train_model(data: dict = None, verbose: bool = True):
    config.set_seed()
    device = torch.device(config.DEVICE if torch.cuda.is_available() or config.DEVICE == "cpu" else "cpu")

    if data is None:
        data = prepare_data()

    train_loader, val_loader = make_loaders(data)

    model = GRUForecaster(
        n_features=data["n_features"],
        hidden_size=config.GRU_HIDDEN_SIZE,
        num_layers=config.GRU_NUM_LAYERS,
        dropout=config.GRU_DROPOUT,
        bidirectional=config.GRU_BIDIRECTIONAL,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8
    )

    best_val_loss = np.inf
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}
    best_state = None

    start_time = time.time()
    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        improved = val_loss < best_val_loss - 1e-6
        if improved:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(
                f"Epoch {epoch:4d}/{config.NUM_EPOCHS} | "
                f"train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | "
                f"best_val_loss={best_val_loss:.6f} | "
                f"no_improve={epochs_no_improve}"
            )

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            if verbose:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {config.EARLY_STOPPING_PATIENCE} epochs).")
            break

    elapsed = time.time() - start_time

    if best_state is not None:
        model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(config.MODEL_CHECKPOINT_PATH), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": data["n_features"],
            "hidden_size": config.GRU_HIDDEN_SIZE,
            "num_layers": config.GRU_NUM_LAYERS,
            "dropout": config.GRU_DROPOUT,
            "bidirectional": config.GRU_BIDIRECTIONAL,
        },
        config.MODEL_CHECKPOINT_PATH,
    )

    history_path = os.path.join(config.OUTPUT_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    if verbose:
        print(f"\nTraining finished in {elapsed:.1f}s. "
              f"Best val_loss={best_val_loss:.6f}")
        print(f"Model checkpoint saved to: {config.MODEL_CHECKPOINT_PATH}")
        print(f"Training history saved to: {history_path}")

    return model, history, data, device


if __name__ == "__main__":
    model, history, data, device = train_model(verbose=True)

    # Quick sanity forecast on the val set (1-step-ahead), converted back to
    # physical units via the fitted target scaler.
    model.eval()
    with torch.no_grad():
        X_val = torch.from_numpy(data["X_val"]).to(device)
        preds_scaled = model(X_val).cpu().numpy()

    preds = data["target_scaler"].inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    y_true = data["target_scaler"].inverse_transform(data["y_val"].reshape(-1, 1)).flatten()

    val_mape = mape_fn(y_true, preds)
    print(f"\nValidation MAPE (1-step-ahead, physical units): {val_mape:.2f}%")
    assert not np.isnan(val_mape)
    print("Training script sanity check passed.")
