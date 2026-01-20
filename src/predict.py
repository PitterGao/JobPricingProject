from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.config import Paths, FeatureConfig
from src.utils import load_joblib, device


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


def _fill_defaults(row: Dict[str, Any]) -> Dict[str, Any]:
    defaults = {
        "training_budget": 60.0,
        "promotion_rate": 55.0,
        "work_life_balance": 58.0,
        "hr_response_speed": 62.0,
        "benefits_score": 57.0,
        "intl_flag": 0,
        "target_top10": 0,
        "funding_stage": "B",
        "enterprise_tier": "T2",
        "location": "sydney",
        "job_level": "mid",
        "job_function": "backend",
        "salary_min": 90000,
        "salary_max": 130000,
        "job_health_score": 0.5,
        "company_size": 300,
        "brand_level": 3,
    }
    out = dict(defaults)
    out.update({k: v for k, v in row.items() if v is not None})
    out["salary_mid"] = int((float(out["salary_min"]) + float(out["salary_max"])) / 2.0)
    return out


def _compute_talent_care_index(row: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
    soft_cols = ["training_budget", "promotion_rate", "work_life_balance", "hr_response_speed", "benefits_score"]
    soft = np.array([float(row[c]) for c in soft_cols], dtype=float)
    soft_scaled = np.clip(soft / 100.0, 0.0, 1.0)
    weights = np.array([0.25, 0.20, 0.20, 0.15, 0.20], dtype=float)

    tci = float((soft_scaled * weights).sum())

    breakdown = {
        "training_budget": float(row["training_budget"]) / 100.0 * 0.25,
        "promotion_rate": float(row["promotion_rate"]) / 100.0 * 0.20,
        "work_life_balance": float(row["work_life_balance"]) / 100.0 * 0.20,
        "hr_response_speed": float(row["hr_response_speed"]) / 100.0 * 0.15,
        "benefits_score": float(row["benefits_score"]) / 100.0 * 0.20,
    }
    return tci, breakdown


def _predict_price_from_mlp(
        mlp: nn.Module,
        Xt_price: np.ndarray,
        price_target_transform: str,
        clip_low: float = 30.0,
        clip_high: float = 5000.0,
) -> float:
    dev = device()
    with torch.no_grad():
        raw = float(mlp(torch.tensor(Xt_price, dtype=torch.float32).to(dev)).cpu().numpy()[0])

    if price_target_transform == "log1p":
        pred = float(np.expm1(raw))
    else:
        pred = float(raw)

    pred = float(np.clip(pred, clip_low, clip_high))
    return pred


# Public APIs
def load_predictor(price_tag: str = "none"):
    """
    price_tag: none / lstm / transformer
    """
    paths = Paths()
    feat_cfg = FeatureConfig()

    enc = load_joblib(paths.models / "encoders.pkl")
    transformer = enc["transformer"]

    tier_model = load_joblib(paths.models / "tier_model.pkl")
    xgb = load_joblib(paths.models / "impression_xgb.pkl")

    # price model
    mlp_ckpt = load_joblib(paths.models / f"price_mlp_{price_tag}.pkl")

    dev = device()
    mlp = PriceMLP(
        in_dim=mlp_ckpt["in_dim"],
        h1=mlp_ckpt["h1"],
        h2=mlp_ckpt["h2"],
        dropout=mlp_ckpt["dropout"],
    ).to(dev)
    mlp.load_state_dict(mlp_ckpt["state_dict"])
    mlp.eval()

    price_target_transform = str(mlp_ckpt.get("target_transform", "none"))

    return {
        "transformer": transformer,
        "tier_model": tier_model,
        "xgb": xgb,
        "mlp": mlp,
        "feature_cfg": feat_cfg,
        "price_target_transform": price_target_transform,
        "price_tag": price_tag,
    }


def predict_one(predictor: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    feat_cfg: FeatureConfig = predictor["feature_cfg"]
    transformer = predictor["transformer"]
    tier_model = predictor["tier_model"]
    xgb = predictor["xgb"]
    mlp = predictor["mlp"]
    price_target_transform = predictor.get("price_target_transform", "none")

    row = _fill_defaults(payload)

    tci, breakdown = _compute_talent_care_index(row)
    row["talent_care_index"] = float(tci)

    stage_map = {"seed": 1, "A": 2, "B": 3, "C": 4, "D": 5, "public": 6}
    funding_stage_num = stage_map.get(str(row["funding_stage"]), 3)

    company_df = pd.DataFrame([{
        "company_size": int(row.get("company_size", 300)),
        "brand_level": int(row.get("brand_level", 3)),
        "funding_stage_num": int(funding_stage_num),
        "talent_care_index": float(row["talent_care_index"]),
    }])
    tier = tier_model.predict_tier(company_df).iloc[0]
    row["enterprise_tier"] = tier

    # Predict impressions (XGB, log1p domain)
    row_imp = dict(row)
    row_imp["pred_impressions"] = 0.0  # consistent with training/inference
    # ensure health exists
    row_imp["job_health_score"] = float(row_imp.get("job_health_score", 0.5))

    X_imp = pd.DataFrame([row_imp])
    X_imp = X_imp[list(feat_cfg.categorical_cols) + list(feat_cfg.numeric_cols)]
    Xt_imp = transformer.transform(X_imp)

    pred_log = float(xgb.predict(Xt_imp)[0])
    pred_impressions = float(np.expm1(pred_log))
    pred_impressions = float(max(0.0, pred_impressions))

    # Predict price (MLP)
    row_price = dict(row)
    row_price["pred_impressions"] = pred_impressions
    row_price["job_health_score"] = float(row_price.get("job_health_score", 0.5))

    X_price = pd.DataFrame([row_price])
    X_price = X_price[list(feat_cfg.categorical_cols) + list(feat_cfg.numeric_cols)]
    Xt_price = transformer.transform(X_price).astype(np.float32)

    price_pred = _predict_price_from_mlp(
        mlp=mlp,
        Xt_price=Xt_price,
        price_target_transform=price_target_transform,
        clip_low=30.0,
        clip_high=5000.0,
    )

    return {
        "enterprise_tier": tier,
        "talent_care_index": float(row["talent_care_index"]),
        "talent_care_breakdown": breakdown,
        "pred_impressions": pred_impressions,
        "pred_price": price_pred,
        "price_model_tag": predictor.get("price_tag", "none"),
        "price_target_transform": price_target_transform,
    }
