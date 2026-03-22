# -*- coding: utf-8 -*-
"""
Train AERIS NN models on the full database.

Produces two checkpoints:
  1) model/aeris_comp_only.pt     -- composition + Magpie (230 features)
  2) model/aeris_full_struct.pt   -- composition + Magpie + structural (236 features, same as RF)

Both use the AerisFullStructure architecture from aeris.py.
Training on data/Dataset_feature+CN.csv (48,490 entries).

Usage:
  python training/train_nn.py                  # train both
  python training/train_nn.py --mode comp      # comp-only
  python training/train_nn.py --mode struct    # full-structure
  python training/train_nn.py --epochs 200     # custom epochs
"""

import sys, os, argparse, time, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# resolve project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from aeris import AerisFullStructure

# ---- Feature definitions (matching RF model_3.joblib exactly) ----------------

# Load RF feature names as ground truth
_rf_model = joblib.load("model/model_3.joblib")
RF_FEATURES = list(_rf_model.feature_names_in_)  # 236 features
del _rf_model

STRUCT_COLS = ["CN_max", "CN_min", "CN_avg", "nsites", "spacegroup_number", "nelements"]
COMP_FEATURES = [f for f in RF_FEATURES if f not in STRUCT_COLS]  # 230 features

print(f"Feature sets: RF={len(RF_FEATURES)}, comp-only={len(COMP_FEATURES)}, struct={len(STRUCT_COLS)}")


def load_data(feature_names, target_col="formation_energy_per_atom"):
    """Load dataset and build X, y arrays in feature_names order."""
    print("Loading data/Dataset_feature+CN.csv ...")
    db = pd.read_csv("data/Dataset_feature+CN.csv", low_memory=False)
    print(f"  Raw entries: {len(db)}")

    # Drop rows with missing target
    db = db.dropna(subset=[target_col])
    print(f"  With valid target: {len(db)}")

    # Build X matrix in exact feature order
    n = len(db)
    nf = len(feature_names)
    X = np.zeros((n, nf), dtype=np.float32)

    for j, fname in enumerate(feature_names):
        if fname in db.columns:
            col = pd.to_numeric(db[fname], errors="coerce").fillna(0.0).values
            X[:, j] = col.astype(np.float32)
        else:
            X[:, j] = 0.0

    y = db[target_col].values.astype(np.float32)

    # Clean
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    mask = np.isfinite(y)
    X, y = X[mask], y[mask]
    print(f"  Final: {len(y)} samples, {nf} features")
    return X, y


def train_model(X, y, feature_names, out_path, epochs=100, batch_size=512, lr=1e-3,
                val_frac=0.1, patience=15):
    """Train AerisFullStructure NN and save checkpoint."""
    nf = X.shape[1]
    print(f"\n{'='*65}")
    print(f"Training: {out_path}")
    print(f"  Features: {nf}, Samples: {len(y)}, Epochs: {epochs}")
    print(f"{'='*65}")

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_frac, random_state=42
    )
    print(f"  Train: {len(y_train)}, Val: {len(y_val)}")

    # Scale features
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s = scaler.transform(X_val).astype(np.float32)

    # DataLoaders
    train_ds = TensorDataset(
        torch.tensor(X_train_s), torch.tensor(y_train).unsqueeze(1)
    )
    val_ds = TensorDataset(
        torch.tensor(X_val_s), torch.tensor(y_val).unsqueeze(1)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    model = AerisFullStructure(input_dim=nf, dropout=0.3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    epochs_no_improve = 0
    history = []

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_loss = 0.0
        n_train = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            n_train += len(xb)
        train_loss /= n_train

        # Validate
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                pred = model(xb).cpu().numpy().flatten()
                val_preds.extend(pred)
                val_true.extend(yb.numpy().flatten())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)
        val_mae = float(np.mean(np.abs(val_preds - val_true)))
        val_rmse = float(np.sqrt(np.mean((val_preds - val_true) ** 2)))
        scheduler.step(val_mae)

        history.append({
            "epoch": epoch, "train_mse": train_loss,
            "val_mae": val_mae, "val_rmse": val_rmse
        })

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            marker = " *"
        else:
            epochs_no_improve += 1
            marker = ""

        if epoch % 10 == 0 or epoch == 1 or marker:
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train_mse={train_loss:.5f}  "
                  f"val_MAE={val_mae:.4f}  val_RMSE={val_rmse:.4f}  "
                  f"lr={lr_now:.1e}  [{elapsed:.0f}s]{marker}")

        if epochs_no_improve >= patience:
            print(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Save best model
    model.load_state_dict(best_state)
    model.eval()

    # Final metrics on val
    val_preds_final = []
    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device)
            val_preds_final.extend(model(xb).cpu().numpy().flatten())
    val_preds_final = np.array(val_preds_final)
    final_mae = float(np.mean(np.abs(val_preds_final - val_true)))
    final_rmse = float(np.sqrt(np.mean((val_preds_final - val_true) ** 2)))

    ckpt = {
        "model_state_dict": best_state,
        "input_dim": nf,
        "feature_names": list(feature_names),
        "scaler": scaler,
        "metrics": {"val_mae": final_mae, "val_rmse": final_rmse},
        "history": history,
    }
    torch.save(ckpt, out_path)
    print(f"\n  Saved: {out_path}")
    print(f"  Best val MAE: {final_mae:.4f} eV/atom")
    print(f"  Best val RMSE: {final_rmse:.4f} eV/atom")
    print(f"  Total time: {time.time() - t0:.0f}s")
    return final_mae, final_rmse


def main():
    parser = argparse.ArgumentParser(description="Train AERIS NN models")
    parser.add_argument("--mode", choices=["comp", "struct", "both"], default="both",
                        help="comp=composition-only, struct=full-structure, both=train both")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()

    if args.mode in ("comp", "both"):
        print("\n" + "=" * 65)
        print("  COMPOSITION-ONLY MODEL (230 features)")
        print("=" * 65)
        X, y = load_data(COMP_FEATURES)
        train_model(X, y, COMP_FEATURES, "model/aeris_comp_only.pt",
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, patience=args.patience)

    if args.mode in ("struct", "both"):
        print("\n" + "=" * 65)
        print("  FULL-STRUCTURE MODEL (236 features, same as RF)")
        print("=" * 65)
        X, y = load_data(RF_FEATURES)
        train_model(X, y, RF_FEATURES, "model/aeris_full_struct.pt",
                    epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, patience=args.patience)


if __name__ == "__main__":
    main()
