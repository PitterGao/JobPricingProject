from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

from src.config import Paths, DataConfig, TierConfig, FeatureConfig
from src.utils import ensure_dirs, set_seed, save_joblib
from src.tier_model import TierModelArtifact
from src.config import HealthConfig

# Synthetic data generation
JOB_FUNCTIONS = ["backend", "frontend", "data", "algo", "product", "qa", "ops"]
JOB_LEVELS = ["junior", "mid", "senior", "lead"]
LOCATIONS = ["sydney", "melbourne", "remote", "shanghai", "beijing"]
FUNDING_STAGES = ["seed", "A", "B", "C", "D", "public"]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def hot_function(job_function: str) -> int:
    return 1 if job_function in {"algo", "data", "backend"} else 0

def generate_company_table(cfg: DataConfig, tier_cfg: TierConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    company_id = np.arange(1, cfg.n_companies + 1)

    company_size = rng.integers(20, 5000, size=cfg.n_companies)
    brand_level = rng.integers(1, 6, size=cfg.n_companies)  # 1..5
    funding_stage = rng.choice(FUNDING_STAGES, size=cfg.n_companies, p=[0.12,0.18,0.18,0.16,0.10,0.26])
    intl_flag = rng.integers(0, 2, size=cfg.n_companies)

    # soft metrics (0-100-ish)
    training_budget = np.clip(rng.normal(60, 15, cfg.n_companies), 0, 100)
    promotion_rate = np.clip(rng.normal(55, 18, cfg.n_companies), 0, 100)
    work_life_balance = np.clip(rng.normal(58, 16, cfg.n_companies), 0, 100)
    hr_response_speed = np.clip(rng.normal(62, 14, cfg.n_companies), 0, 100)
    benefits_score = np.clip(rng.normal(57, 17, cfg.n_companies), 0, 100)

    df = pd.DataFrame({
        "company_id": company_id,
        "company_size": company_size,
        "brand_level": brand_level,
        "funding_stage": funding_stage,
        "intl_flag": intl_flag,
        "training_budget": training_budget,
        "promotion_rate": promotion_rate,
        "work_life_balance": work_life_balance,
        "hr_response_speed": hr_response_speed,
        "benefits_score": benefits_score,
    })

    stage_map = {"seed": 1, "A": 2, "B": 3, "C": 4, "D": 5, "public": 6}
    df["funding_stage_num"] = df["funding_stage"].map(stage_map).astype(int)

    soft_cols = list(tier_cfg.weights.keys())
    scaler = MinMaxScaler()
    soft_scaled = scaler.fit_transform(df[soft_cols])
    w = np.array([tier_cfg.weights[c] for c in soft_cols], dtype=float)
    df["talent_care_index"] = (soft_scaled * w).sum(axis=1)

    return df

def generate_job_table(cfg: DataConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed + 1)
    job_id = np.arange(1, cfg.n_jobs + 1)
    company_id = rng.integers(1, cfg.n_companies + 1, size=cfg.n_jobs)

    job_function = rng.choice(JOB_FUNCTIONS, size=cfg.n_jobs)
    job_level = rng.choice(JOB_LEVELS, size=cfg.n_jobs, p=[0.25, 0.40, 0.25, 0.10])
    location = rng.choice(LOCATIONS, size=cfg.n_jobs)

    level_base = {"junior": 60_000, "mid": 95_000, "senior": 140_000, "lead": 190_000}
    func_premium = {"backend": 1.05, "frontend": 1.0, "data": 1.12, "algo": 1.18, "product": 1.0, "qa": 0.92, "ops": 0.98}
    base = np.array([level_base[l] for l in job_level]) * np.array([func_premium[f] for f in job_function])
    noise = rng.normal(0, 12_000, size=cfg.n_jobs)
    salary_mid = np.clip(base + noise, 45_000, 320_000)
    salary_min = np.clip(salary_mid * rng.uniform(0.85, 0.93, size=cfg.n_jobs), 40_000, None)
    salary_max = np.clip(salary_mid * rng.uniform(1.07, 1.20, size=cfg.n_jobs), None, 380_000)

    target_top10 = rng.integers(0, 2, size=cfg.n_jobs)

    df = pd.DataFrame({
        "job_id": job_id,
        "company_id": company_id,
        "job_function": job_function,
        "job_level": job_level,
        "location": location,
        "salary_min": salary_min.astype(int),
        "salary_max": salary_max.astype(int),
        "salary_mid": salary_mid.astype(int),
        "target_top10": target_top10,
    })
    return df

def simulate_labels(company_df: pd.DataFrame, job_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 2)

    merged = job_df.merge(company_df, on="company_id", how="left")

    hf = merged["job_function"].map(hot_function).astype(int)
    attract_score = (
        0.6 * (merged["brand_level"] - 1) +
        0.00001 * merged["salary_mid"] +
        0.8 * merged["intl_flag"] +
        0.7 * merged["target_top10"] +
        0.9 * hf +
        1.2 * merged["talent_care_index"] +
        rng.normal(0, 0.6, size=len(merged))
    )
    p_view = sigmoid(attract_score)
    lam = np.exp(3.6 + 3.0 * (p_view - 0.5))
    impressions = rng.poisson(lam)

    apply_rate = np.clip(0.015 + 0.04 * p_view, 0.01, 0.08)
    apply_cnt = rng.binomial(impressions, apply_rate)

    topschoolratio = np.clip(
        0.08 + 0.03 * (merged["brand_level"] - 1) + 0.05 * merged["target_top10"] + 0.07 * merged["talent_care_index"]
        + rng.normal(0, 0.02, size=len(merged)),
        0.02, 0.45
    )

    expected_applies = apply_cnt * topschoolratio
    value_per_hq_apply = 600.0
    expected_value = value_per_hq_apply * expected_applies

    roi_target = rng.uniform(1.1, 2.2, size=len(merged))
    # roi_target = 3
    brand_factor = 0.8 + 0.1 * merged["brand_level"]
    price_label = expected_value / roi_target * brand_factor
    # price_label = np.clip(price_label, 100, 5000)
    price_label = np.clip(price_label, 30, 5000)

    out = merged.copy()
    out["impressions"] = impressions
    out["apply_cnt"] = apply_cnt
    out["topschoolratio"] = topschoolratio
    out["expected_applies"] = expected_applies
    out["price_label"] = price_label.astype(float)
    return out

def generate_job_apply_logs(train_df: pd.DataFrame, health_cfg: HealthConfig, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 100)

    rows = []
    total_days = health_cfg.total_days

    for r in train_df.itertuples(index=False):
        job_id = int(r.job_id)
        impressions_total = float(r.impressions)
        apply_total = float(getattr(r, "apply_cnt", max(1.0, impressions_total * 0.03)))
        topschoolratio = float(getattr(r, "topschoolratio", 0.12))

        w = rng.lognormal(mean=0.0, sigma=0.35, size=total_days)
        w = w / w.sum()

        imp_days = rng.poisson(lam=np.maximum(0.1, impressions_total * w))
        app_days = np.array([rng.binomial(int(imp), p=min(0.2, max(0.01, apply_total / max(impressions_total, 1.0)))) for imp in imp_days])

        hq_days = np.array([rng.binomial(int(a), p=float(np.clip(topschoolratio + rng.normal(0, 0.01), 0.02, 0.5))) for a in app_days])

        success = np.zeros_like(app_days, dtype=float)
        np.divide(hq_days, app_days, out=success, where=app_days > 0)

        for d in range(total_days):
            rows.append({
                "job_id": job_id,
                "day": d,
                "impressions": int(imp_days[d]),
                "apply_cnt": int(app_days[d]),
                "hq_apply_cnt": int(hq_days[d]),
                "success_rate": float(success[d]),
            })

    return pd.DataFrame(rows)


def build_health_sequences(logs_df: pd.DataFrame, health_cfg: HealthConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = health_cfg.history_len
    H = health_cfg.horizon

    xs, ys, jids = [], [], []
    for job_id, g in logs_df.groupby("job_id"):
        g = g.sort_values("day")
        if len(g) < T + H:
            continue

        hist = g.iloc[:T]
        fut = g.iloc[T:T+H]

        feat = hist[["impressions", "apply_cnt", "success_rate"]].to_numpy(dtype=np.float32)
        label = float(fut["success_rate"].mean())

        feat[:, 0] = np.log1p(feat[:, 0])
        feat[:, 1] = np.log1p(feat[:, 1])

        xs.append(feat)
        ys.append(label)
        jids.append(int(job_id))

    X = np.stack(xs, axis=0) if xs else np.zeros((0, T, 3), dtype=np.float32)
    y = np.array(ys, dtype=np.float32) if ys else np.zeros((0,), dtype=np.float32)
    job_ids = np.array(jids, dtype=np.int64) if jids else np.zeros((0,), dtype=np.int64)
    return X, y, job_ids


# Build Tier model artifact
def fit_tier_model(company_df: pd.DataFrame, seed: int) -> TierModelArtifact:
    hard_cols = ["company_size", "brand_level", "funding_stage_num"]
    scaler = StandardScaler()
    Xh = scaler.fit_transform(company_df[hard_cols])

    kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
    kmeans.fit(Xh)

    q_low = float(company_df["talent_care_index"].quantile(0.2))
    q_high = float(company_df["talent_care_index"].quantile(0.8))

    return TierModelArtifact(
        hard_scaler=scaler,
        kmeans=kmeans,
        q_low=q_low,
        q_high=q_high
    )


def build_transformer(feature_cfg: FeatureConfig) -> ColumnTransformer:
    cat_cols = list(feature_cfg.categorical_cols)
    num_cols = list(feature_cfg.numeric_cols)

    transformer = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", StandardScaler(), num_cols),
        ],
        remainder="drop"
    )
    return transformer

def main():
    paths = Paths()
    data_cfg = DataConfig()
    tier_cfg = TierConfig()
    feat_cfg = FeatureConfig()
    health_cfg = HealthConfig()

    ensure_dirs(paths.data_raw, paths.data_processed, paths.models)
    set_seed(data_cfg.seed)

    company_df = generate_company_table(data_cfg, tier_cfg)
    job_df = generate_job_table(data_cfg)

    tier_art = fit_tier_model(company_df, data_cfg.seed)
    company_df["enterprise_tier"] = tier_art.predict_tier(company_df)

    full_df = simulate_labels(company_df, job_df, data_cfg.seed)

    full_df["pred_impressions"] = full_df["impressions"].astype(float)
    full_df["job_health_score"] = 0.5

    # ---------- bonus: generate job_apply_logs & health sequences ----------
    logs_df = generate_job_apply_logs(full_df, health_cfg, data_cfg.seed)
    logs_df.to_parquet(paths.data_processed / "job_apply_logs.parquet", index=False)

    Xh, yh, job_ids = build_health_sequences(logs_df, health_cfg)
    np.savez_compressed(
        paths.data_processed / "health_sequences.npz",
        X=Xh, y=yh, job_ids=job_ids
    )

    keep_cols = [
        # ids
        "job_id",
        "company_id",
        # company
        "company_size",
        "brand_level",
        "funding_stage",
        "intl_flag",
        "training_budget",
        "promotion_rate",
        "work_life_balance",
        "hr_response_speed",
        "benefits_score",
        "talent_care_index",
        "enterprise_tier",
        # job
        "job_function",
        "job_level",
        "location",
        "salary_min",
        "salary_max",
        "salary_mid",
        "target_top10",
        # labels
        "impressions",
        "price_label",
        # pricing drivers
        "apply_cnt",
        "topschoolratio",
        "expected_applies",
        # helpers / features
        "pred_impressions",
        "job_health_score",
    ]

    train_df = full_df[keep_cols].copy()

    # save raw tables
    company_df.to_csv(paths.data_raw / "company_profile.csv", index=False)
    job_df.to_csv(paths.data_raw / "job_profile.csv", index=False)

    # save train samples
    train_df.to_parquet(paths.data_processed / "train_samples.parquet", index=False)

    X = train_df[list(feat_cfg.categorical_cols) + list(feat_cfg.numeric_cols)].copy()
    transformer = build_transformer(feat_cfg)
    transformer.fit(X)

    artifacts = {
        "transformer": transformer,
        "feature_cfg": feat_cfg,
    }
    save_joblib(artifacts, paths.models / "encoders.pkl")
    save_joblib(tier_art, paths.models / "tier_model.pkl")

    print("Preprocess done.")
    print(f"- train_samples:     {paths.data_processed / 'train_samples.parquet'}")
    print(f"- job_apply_logs:    {paths.data_processed / 'job_apply_logs.parquet'}")
    print(f"- health_sequences:  {paths.data_processed / 'health_sequences.npz'}")
    print(f"- encoders:          {paths.models / 'encoders.pkl'}")
    print(f"- tier_model:        {paths.models / 'tier_model.pkl'}")


if __name__ == "__main__":
    main()

