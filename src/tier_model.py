from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@dataclass
class TierModelArtifact:
    hard_scaler: StandardScaler
    kmeans: KMeans
    q_low: float
    q_high: float

    def predict_tier(self, company_df: pd.DataFrame) -> pd.Series:
        hard_cols = ["company_size", "brand_level", "funding_stage_num"]
        Xh = self.hard_scaler.transform(company_df[hard_cols])
        cluster = self.kmeans.predict(Xh)

        tmp = pd.DataFrame({"cluster": cluster, "hard_score": Xh.mean(axis=1)})
        cluster_rank = (
            tmp.groupby("cluster")["hard_score"].mean().sort_values(ascending=False).index.tolist()
        )
        rank_map = {cluster_rank[i]: i + 1 for i in range(len(cluster_rank))}  # 1..3
        base_rank = pd.Series(cluster).map(rank_map).astype(int)

        tci = company_df["talent_care_index"].astype(float)
        adj_rank = base_rank.copy()
        adj_rank[tci >= self.q_high] = np.maximum(1, adj_rank[tci >= self.q_high] - 1)
        adj_rank[tci <= self.q_low] = np.minimum(3, adj_rank[tci <= self.q_low] + 1)

        return adj_rank.map({1: "T1", 2: "T2", 3: "T3"})
