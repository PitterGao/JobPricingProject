from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.config import Paths, HealthConfig
from src.health_models import HealthLSTM, HealthTransformer
from src.utils import set_seed, device, mae, rmse, save_joblib, save_json

def load_npz(path):
    z = np.load(path)
    return z["X"].astype(np.float32), z["y"].astype(np.float32), z["job_ids"].astype(np.int64)

def build_model(model_name: str, cfg: HealthConfig):
    if model_name == "lstm":
        return HealthLSTM(feat_dim=cfg.feat_dim, hidden=64, num_layers=1, dropout=cfg.dropout)
    if model_name == "transformer":
        return HealthTransformer(feat_dim=cfg.feat_dim, d_model=cfg.d_model, nhead=cfg.nhead, num_layers=cfg.num_layers, dropout=cfg.dropout)
    raise ValueError("health_model must be one of: lstm, transformer")

def train(model, X_train, y_train, X_val, y_val, cfg: HealthConfig):
    dev = device()
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.L1Loss()

    Xtr = torch.tensor(X_train, dtype=torch.float32).to(dev)
    ytr = torch.tensor(y_train, dtype=torch.float32).to(dev)
    Xva = torch.tensor(X_val, dtype=torch.float32).to(dev)
    yva = torch.tensor(y_val, dtype=torch.float32).to(dev)

    n = Xtr.shape[0]
    best = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        idx = torch.randperm(n)
        for start in range(0, n, cfg.batch_size):
            end = start + cfg.batch_size
            b = idx[start:end]
            xb = Xtr[b]
            yb = ytr[b]

            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            pv = model(Xva).detach().cpu().numpy()
        v = float(np.mean(np.abs(pv - y_val)))
        if v < best:
            best = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{cfg.epochs} | val_MAE={best:.4f}")

    return best_state, best

def predict_all(model, state_dict, X):
    dev = device()
    model.load_state_dict(state_dict)
    model.to(dev)
    model.eval()
    with torch.no_grad():
        p = model(torch.tensor(X, dtype=torch.float32).to(dev)).cpu().numpy()
    return p

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--health_model", type=str, default="lstm", choices=["lstm", "transformer"])
    args = parser.parse_args()

    paths = Paths()
    cfg = HealthConfig()
    set_seed(42)

    X, y, job_ids = load_npz(paths.data_processed / "health_sequences.npz")
    if len(X) == 0:
        raise RuntimeError("health_sequences.npz is empty. Please re-run preprocess.")

    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model(args.health_model, cfg)
    best_state, best_mae = train(model, X_tr, y_tr, X_va, y_va, cfg)

    # metrics
    pred_va = predict_all(build_model(args.health_model, cfg), best_state, X_va)
    metrics = {
        "val_mae": mae(y_va, pred_va),
        "val_rmse": rmse(y_va, pred_va),
        "best_val_mae": float(best_mae),
        "health_model": args.health_model,
    }

    pred_all = predict_all(build_model(args.health_model, cfg), best_state, X)
    pred_all = np.clip(pred_all, 0.0, 1.0)

    ckpt = {
        "health_model": args.health_model,
        "state_dict": best_state,
        "cfg": cfg,
    }
    save_joblib(ckpt, paths.models / f"health_{args.health_model}.pkl")
    save_json(metrics, paths.models / f"metrics_health_{args.health_model}.json")

    job_health = {"job_id": job_ids.tolist(), "job_health_score": pred_all.astype(float).tolist()}
    save_joblib(job_health, paths.models / f"job_health_scores_{args.health_model}.pkl")

    print("Health model saved:", paths.models / f"health_{args.health_model}.pkl")
    print("Job health mapping saved:", paths.models / f"job_health_scores_{args.health_model}.pkl")
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
