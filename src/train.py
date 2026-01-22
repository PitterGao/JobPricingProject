from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from src.config import Paths, FeatureConfig, XGBConfig, MLPConfig
from src.utils import load_joblib, save_joblib, save_json, set_seed, mae, rmse, device


@dataclass
class TrainArtifacts:
    xgb_impression: XGBRegressor
    mlp_state: Dict
    mlp_meta: Dict


class PriceMLP(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_xgb_impressions(X_train, y_train, X_val, y_val, cfg: XGBConfig):
    model = XGBRegressor(**cfg.params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_mlp_price(X_train, y_train, X_val, y_val, in_dim: int, cfg: MLPConfig):
    dev = device()
    model = PriceMLP(in_dim, cfg.hidden1, cfg.hidden2, cfg.dropout).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    # loss_fn = nn.L1Loss()
    # loss_fn = nn.HuberLosds(delta=1.0)
    loss_fn = nn.SmoothL1Loss(beta=0.5, reduction="none")

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.float32)

    # simple mini-batch training
    n = X_train_t.shape[0]
    best_val = float("inf")
    best_state = None

    for epoch in range(cfg.epochs):
        model.train()
        idx = torch.randperm(n)
        Xb = X_train_t[idx].to(dev)
        yb = y_train_t[idx].to(dev)

        # batch loop
        for start in range(0, n, cfg.batch_size):
            end = start + cfg.batch_size
            xb = Xb[start:end]
            ytrue = yb[start:end]

            opt.zero_grad()
            ypred = model(xb)
            per_loss = loss_fn(ypred, ytrue)
            price_true = torch.expm1(ytrue).clamp(min=0.0)
            alpha = 0.8
            # w = 1.0 + alpha * (price_true / (price_true.mean() + 1e-6))
            w = 1.0 + alpha * (price_true / (price_true.mean() + 1e-6)) ** 1.5
            loss = (per_loss * w).mean()
            loss.backward()
            opt.step()

        # val
        model.eval()
        with torch.no_grad():
            ypv = model(X_val_t.to(dev)).cpu().numpy()
        v = float(np.mean(np.abs(ypv - y_val)))
        if v < best_val:
            best_val = v
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:02d}/{cfg.epochs} | val_MAE={best_val:.3f}")

    return best_state, {"best_val_mae": best_val}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--health_model",
        type=str,
        default="none",
        choices=["none", "lstm", "transformer"],
        help="选择接入的健康度模型类型：none/lstm/transformer"
    )
    args = parser.parse_args()

    paths = Paths()
    feat_cfg = FeatureConfig()
    xgb_cfg = XGBConfig()
    mlp_cfg = MLPConfig()

    set_seed(42)

    # Load data
    df = pd.read_parquet(paths.data_processed / "train_samples.parquet")

    if args.health_model != "none":
        mapping_path = paths.models / f"job_health_scores_{args.health_model}.pkl"
        mapping = load_joblib(mapping_path)
        map_df = pd.DataFrame(mapping)[["job_id", "job_health_score"]]

        if "job_health_score" in df.columns:
            df = df.drop(columns=["job_health_score"])

        df = df.merge(map_df, on="job_id", how="left")
        df["job_health_score"] = df["job_health_score"].fillna(0.5)
        print(f"Loaded health scores: {mapping_path}")
    else:
        df["job_health_score"] = df.get("job_health_score", 0.5)
        df["job_health_score"] = df["job_health_score"].fillna(0.5)

    # encoders/transformer
    enc = load_joblib(paths.models / "encoders.pkl")
    transformer = enc["transformer"]

    X_cols = list(feat_cfg.categorical_cols) + list(feat_cfg.numeric_cols)

    idx_all = np.arange(len(df))
    idx_train, idx_val = train_test_split(idx_all, test_size=0.2, random_state=42)

    # Train impression model (XGB)
    df_imp = df.copy()
    df_imp["pred_impressions"] = 0.0
    if "job_health_score" in df_imp.columns:
        df_imp["job_health_score"] = df_imp["job_health_score"].fillna(0.5)

    X_imp = df_imp[X_cols]
    y_imp = np.log1p(df_imp["impressions"].values.astype(float))

    X_train = X_imp.iloc[idx_train]
    X_val = X_imp.iloc[idx_val]
    y_train = y_imp[idx_train]
    y_val = y_imp[idx_val]

    X_train_t = transformer.transform(X_train)
    X_val_t = transformer.transform(X_val)

    xgb = train_xgb_impressions(X_train_t, y_train, X_val_t, y_val, xgb_cfg)

    y_pred_val = xgb.predict(X_val_t)

    y_val_raw = np.expm1(y_val)
    y_pred_raw = np.maximum(0.0, np.expm1(y_pred_val))

    metrics_imp = {
        "rmse_log1p": rmse(y_val, y_pred_val),
        "mae_log1p": mae(y_val, y_pred_val),
        "rmse": rmse(y_val_raw, y_pred_raw),
        "mae": mae(y_val_raw, y_pred_raw),
    }

    metrics_imp = {
        "rmse_log1p": rmse(y_val, y_pred_val),
        "mae_log1p": mae(y_val, y_pred_val),
    }

    save_joblib(xgb, paths.models / "impression_xgb.pkl")
    save_json(metrics_imp, paths.models / "metrics_impression.json")
    print("Saved impression_xgb.pkl")

    # Build pred_impressions for pricing model using trained XGB
    df_price = df.copy()

    df_for_pred = df_price.copy()
    df_for_pred["pred_impressions"] = 0.0
    df_for_pred["job_health_score"] = df_for_pred["job_health_score"].fillna(0.5)

    X_all_imp = transformer.transform(df_for_pred[X_cols])
    pred_log_all = xgb.predict(X_all_imp)
    df_price["pred_impressions"] = np.maximum(0.0, np.expm1(pred_log_all))

    df_price["job_health_score"] = df_price["job_health_score"].fillna(0.5)

    # Train pricing model (2-layer MLP)
    Xp = df_price[X_cols]
    yp = np.log1p(df_price["price_label"].values.astype(float))

    X_train = Xp.iloc[idx_train]
    X_val = Xp.iloc[idx_val]
    y_train = yp[idx_train]
    y_val = yp[idx_val]

    X_train_t = transformer.transform(X_train).astype(np.float32)
    X_val_t = transformer.transform(X_val).astype(np.float32)

    in_dim = X_train_t.shape[1]
    best_state, meta = train_mlp_price(
        X_train_t, y_train, X_val_t, y_val, in_dim, mlp_cfg,
    )

    # final eval
    dev = device()
    model = PriceMLP(in_dim, mlp_cfg.hidden1, mlp_cfg.hidden2, mlp_cfg.dropout).to(dev)
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_val_log = model(torch.tensor(X_val_t, dtype=torch.float32).to(dev)).cpu().numpy()

    metrics_log = {
        "mae_log1p": mae(y_val, pred_val_log),
        "rmse_log1p": rmse(y_val, pred_val_log),
    }

    # ---- log-domain linear calibration: y ≈ a * pred + b ----
    if len(pred_val_log) >= 2 and np.std(pred_val_log) > 1e-8:
        a, b = np.polyfit(pred_val_log, y_val, deg=1)
    else:
        a, b = 1.0, 0.0
    pred_val_log = a * pred_val_log + b

    y_val_price = np.expm1(y_val)
    pred_val_price = np.expm1(pred_val_log)
    pred_val_price = np.clip(pred_val_price, 30.0, 5000.0)

    metrics_price = {
        "mae": mae(y_val_price, pred_val_price),
        "rmse": rmse(y_val_price, pred_val_price),
        **metrics_log,
        **meta,
        "health_model": args.health_model,
        "target_transform": "log1p",
    }

    ckpt = {
        "state_dict": best_state,
        "in_dim": in_dim,
        "h1": mlp_cfg.hidden1,
        "h2": mlp_cfg.hidden2,
        "dropout": mlp_cfg.dropout,
        "health_model": args.health_model,
        "target_transform": "log1p",
        "log_calib": {"a": float(a), "b": float(b)},
    }

    tagged_path = paths.models / f"price_mlp_{args.health_model}.pkl"
    save_joblib(ckpt, tagged_path)
    save_json(metrics_price, paths.models / f"metrics_price_{args.health_model}.json")
    print(f"Saved {tagged_path.name}")

    save_joblib(ckpt, paths.models / "price_mlp.pkl")

    print("Done. Metrics:")
    print({"impression": metrics_imp, "price": metrics_price})


if __name__ == "__main__":
    main()
