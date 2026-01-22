from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split

from src.config import Paths, FeatureConfig
from src.utils import load_joblib, mae, rmse, set_seed, device


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


def ensure_reports_dir(paths: Paths) -> Path:
    reports = paths.root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    return reports


def summarize_distribution(name: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=float)
    q = np.quantile(arr, [0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0])
    print(f"\n[{name}] distribution:")
    print("min,1%,5%,50%,95%,99%,max =", q)


def load_price_model(paths: Paths, tag: str):
    ckpt_path = paths.models / f"price_mlp_{tag}.pkl"
    if not ckpt_path.exists():
        return None, None, None

    ckpt = load_joblib(ckpt_path)

    dev = device()
    model = PriceMLP(
        in_dim=ckpt["in_dim"],
        h1=ckpt["h1"],
        h2=ckpt["h2"],
        dropout=ckpt["dropout"],
    ).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    target_transform = ckpt.get("target_transform", "raw")
    return model, ckpt_path, target_transform


def predict_price_batch(model: nn.Module, X_np: np.ndarray) -> np.ndarray:
    dev = device()
    with torch.no_grad():
        x = torch.tensor(X_np, dtype=torch.float32).to(dev)
        y = model(x).cpu().numpy()
    return y.astype(float)


def plot_scatter_true_pred(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path):
    plt.figure()
    plt.scatter(y_true, y_pred, s=8)
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(title)

    lo = float(min(y_true.min(), y_pred.min()))
    hi = float(max(y_true.max(), y_pred.max()))
    plt.plot([lo, hi], [lo, hi])
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_error_hist(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path):
    err = y_pred - y_true
    plt.figure()
    plt.hist(err, bins=40)
    plt.xlabel("Prediction Error (pred - true)")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_model_compare(df_metrics: pd.DataFrame, out_path: Path):
    tags = df_metrics["tag"].tolist()
    mae_vals = df_metrics["price_mae"].tolist()
    rmse_vals = df_metrics["price_rmse"].tolist()

    x = np.arange(len(tags))
    width = 0.35

    plt.figure()
    plt.bar(x - width / 2, mae_vals, width, label="MAE")
    plt.bar(x + width / 2, rmse_vals, width, label="RMSE")
    plt.xticks(x, tags)
    plt.ylabel("Error")
    plt.title("Pricing Model Comparison (Test Set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def plot_calibration_by_decile(y_true: np.ndarray, y_pred: np.ndarray, title: str, out_path: Path):
    # bin by true deciles
    qs = np.quantile(y_true, np.linspace(0, 1, 11))
    bins = np.digitize(y_true, qs[1:-1], right=True)  # 0..9

    mean_true = []
    mean_pred = []
    for b in range(10):
        m = bins == b
        if m.sum() == 0:
            mean_true.append(np.nan)
            mean_pred.append(np.nan)
        else:
            mean_true.append(float(np.mean(y_true[m])))
            mean_pred.append(float(np.mean(y_pred[m])))

    plt.figure()
    plt.plot(mean_true, marker="o", label="Mean True")
    plt.plot(mean_pred, marker="o", label="Mean Pred")
    plt.xlabel("Decile bin (by true)")
    plt.ylabel("Mean value")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    set_seed(42)

    paths = Paths()
    reports_dir = ensure_reports_dir(paths)
    feat_cfg = FeatureConfig()

    # Load data
    df = pd.read_parquet(paths.data_processed / "train_samples.parquet")

    # Load encoders & models
    enc = load_joblib(paths.models / "encoders.pkl")
    transformer = enc["transformer"]
    xgb = load_joblib(paths.models / "impression_xgb.pkl")

    X_cols = list(feat_cfg.categorical_cols) + list(feat_cfg.numeric_cols)

    # Split: train/val/test (evaluation only; fixed random)
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)
    df_train, df_val = train_test_split(df_train, test_size=0.25, random_state=42)  # 0.6/0.2/0.2

    df_imp_test = df_test.copy()
    df_imp_test["pred_impressions"] = 0.0

    if "job_health_score" not in df_imp_test.columns:
        df_imp_test["job_health_score"] = 0.5
    df_imp_test["job_health_score"] = df_imp_test["job_health_score"].fillna(0.5)

    X_imp_test = transformer.transform(df_imp_test[X_cols])
    y_imp_true = df_imp_test["impressions"].values.astype(float)

    y_imp_true_log = np.log1p(y_imp_true)
    y_imp_pred_log = xgb.predict(X_imp_test)
    y_imp_pred = np.maximum(0.0, np.expm1(y_imp_pred_log))

    imp_metrics = {
        "impressions_mae_log1p": mae(y_imp_true_log, y_imp_pred_log),
        "impressions_rmse_log1p": rmse(y_imp_true_log, y_imp_pred_log),
        "impressions_mae_raw": mae(y_imp_true, y_imp_pred),
        "impressions_rmse_raw": rmse(y_imp_true, y_imp_pred),
        "n_test": int(len(y_imp_true)),
    }
    print("\nImpressions metrics on TEST:", imp_metrics)
    summarize_distribution("impressions_true", y_imp_true)
    summarize_distribution("impressions_pred", y_imp_pred)

    plot_scatter_true_pred(
        y_imp_true,
        y_imp_pred,
        "Impressions: True vs Pred (Test)",
        reports_dir / "impressions_true_vs_pred.png",
    )
    plot_error_hist(
        y_imp_true,
        y_imp_pred,
        "Impressions Error Histogram (Test) [raw]",
        reports_dir / "impressions_error_hist.png",
    )
    plot_calibration_by_decile(
        y_imp_true,
        y_imp_pred,
        "Impressions Calibration by True Decile (Test)",
        reports_dir / "impressions_calibration_decile.png",
    )

    pd.DataFrame([imp_metrics]).to_csv(reports_dir / "impressions_metrics.csv", index=False)
    print("Saved:", reports_dir / "impressions_metrics.csv")

    df_price_test = df_test.copy()

    if "job_health_score" not in df_price_test.columns:
        df_price_test["job_health_score"] = 0.5
    df_price_test["job_health_score"] = df_price_test["job_health_score"].fillna(0.5)

    df_for_pred = df_price_test.copy()
    df_for_pred["pred_impressions"] = 0.0
    Xt_all = transformer.transform(df_for_pred[X_cols])
    pred_log = xgb.predict(Xt_all)
    df_price_test["pred_impressions"] = np.maximum(0.0, np.expm1(pred_log))

    Xt_price = transformer.transform(df_price_test[X_cols]).astype(np.float32)
    y_price_true = df_price_test["price_label"].values.astype(float)

    print("\nPricing label stats on TEST:")
    summarize_distribution("price_true", y_price_true)

    y_train_price = df_train["price_label"].values.astype(float)
    naive_pred = np.full_like(y_price_true, fill_value=float(np.median(y_train_price)))
    naive_row = {
        "tag": "naive_median",
        "target_transform": "raw",
        "price_mae": mae(y_price_true, naive_pred),
        "price_rmse": rmse(y_price_true, naive_pred),
        "price_mae_log1p": mae(np.log1p(y_price_true), np.log1p(naive_pred)),
        "price_rmse_log1p": rmse(np.log1p(y_price_true), np.log1p(naive_pred)),
        "n_test": int(len(y_price_true)),
        "model_path": "",
    }
    print("\n[naive_median] price metrics on TEST:", naive_row)

    tags = ["none", "lstm", "transformer"]
    rows: List[Dict] = [naive_row]

    clip_lo = float(min(30.0, np.min(y_train_price)))
    clip_hi = float(max(5000.0, np.max(y_train_price) * 3.0))

    for tag in tags:
        model, ckpt_path, target_transform = load_price_model(paths, tag)
        if model is None:
            print(f"⚠️ Skip pricing model '{tag}': not found models/price_mlp_{tag}.pkl")
            continue

        raw_out = predict_price_batch(model, Xt_price)

        if str(target_transform).lower() == "log1p":
            y_pred = np.expm1(raw_out)
            y_pred = np.maximum(0.0, y_pred)
            y_pred = np.clip(y_pred, clip_lo, clip_hi)

            row = {
                "tag": tag,
                "target_transform": "log1p",
                "price_mae": mae(y_price_true, y_pred),
                "price_rmse": rmse(y_price_true, y_pred),
                "price_mae_log1p": mae(np.log1p(y_price_true), raw_out),
                "price_rmse_log1p": rmse(np.log1p(y_price_true), raw_out),
                "n_test": int(len(y_price_true)),
                "model_path": str(ckpt_path),
            }
        else:
            y_pred = np.clip(raw_out, clip_lo, clip_hi)
            row = {
                "tag": tag,
                "target_transform": "raw",
                "price_mae": mae(y_price_true, y_pred),
                "price_rmse": rmse(y_price_true, y_pred),
                "price_mae_log1p": mae(np.log1p(y_price_true), np.log1p(y_pred)),
                "price_rmse_log1p": rmse(np.log1p(y_price_true), np.log1p(y_pred)),
                "n_test": int(len(y_price_true)),
                "model_path": str(ckpt_path),
            }

        rows.append(row)
        print(f"\n[{tag}] price metrics on TEST:", row)

        plot_scatter_true_pred(
            y_price_true,
            y_pred,
            f"Price: True vs Pred (Test) [{tag}]",
            reports_dir / f"price_true_vs_pred_{tag}.png",
        )
        plot_error_hist(
            y_price_true,
            y_pred,
            f"Price Error Histogram (Test) [{tag}]",
            reports_dir / f"price_error_hist_{tag}.png",
        )
        plot_calibration_by_decile(
            y_price_true,
            y_pred,
            f"Price Calibration by True Decile (Test) [{tag}]",
            reports_dir / f"price_calibration_decile_{tag}.png",
        )

    df_metrics = pd.DataFrame(rows).sort_values("tag")
    df_metrics.to_csv(reports_dir / "metrics_summary.csv", index=False)
    print("\nSaved metrics summary:", reports_dir / "metrics_summary.csv")

    df_plot = df_metrics[df_metrics["tag"] != "naive_median"].copy()
    if len(df_plot) > 0:
        plot_model_compare(df_plot, reports_dir / "model_compare_mae_rmse.png")
        print("Saved plots in:", reports_dir)
    else:
        print("No learned pricing models found to plot compare.")


if __name__ == "__main__":
    main()
