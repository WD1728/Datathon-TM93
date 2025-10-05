# fe.py
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

RANDOM_STATE = 3407

class FeatureEngineer:
    def __init__(self, thr=0.95, random_state=3407):
        self.thr = thr
        self.random_state = random_state
        self.candidate_pairs = [
            ("ALTV", "MLTV"),
            ("ASTV", "ALTV"),
            ("AC", "UC"),
            ("Max", "Min"),
            ("Median", "Mode"),
        ]
        self.qbin_targets = ["ALTV", "ASTV", "Median", "AC", "Mode"]
        self.keep_cols = None
        self.qbin_edges_dict = None
        self.lof = None
        self.iso = None
        self.base_cols = None

    # ========= Core FE Functions =========
    def add_pairwise_ops(self, df):
        df = df.copy()
        for a, b in self.candidate_pairs:
            if a in df.columns and b in df.columns:
                df[f"{a}_over_{b}"] = df[a] / (df[b].replace(0, np.finfo(float).eps))
                df[f"{a}_minus_{b}"] = df[a] - df[b]
                df[f"{a}_times_{b}"] = df[a] * df[b]
        return df

    def add_monotonic(self, df):
        df = df.copy()
        for c in self.base_cols:
            if df[c].min() >= 0:
                df[f"{c}_log1p"] = np.log1p(df[c])
                df[f"{c}_sqrt"] = np.sqrt(df[c])
        return df

    def fit_qbins_edges(self, s, q=5):
        qs = np.linspace(0, 1, q + 1)
        edges = np.unique(np.quantile(s.values, qs))
        if len(edges) <= 2:
            return None
        edges[0] -= 1e-6
        edges[-1] += 1e-6
        return edges

    def cut_with_edges(self, s, edges):
        if edges is None:
            return None
        return pd.cut(s, bins=edges, labels=False, include_lowest=True)

    def fit_outlier_scorers(self, X):
        lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
        lof.fit(X)
        iso = IsolationForest(
            n_estimators=200,
            random_state=self.random_state,
            n_jobs=-1
        ).fit(X)
        return lof, iso

    def transform_outlier_scores(self, df, feat_cols):
        X = df[feat_cols].values
        s_lof = -self.lof.decision_function(X)
        s_iso = -self.iso.decision_function(X)
        out = df.copy()
        out["lof_score"] = s_lof
        out["iforest_abn"] = s_iso
        return out

    def drop_high_corr_cols(self, df):
        corr = df.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [c for c in upper.columns if any(upper[c] > self.thr)]
        keep = [c for c in df.columns if c not in to_drop]
        return keep

    # ========= Fitting =========
    def fit(self, df):
        df = df.copy()
        if "CLASS" in df.columns:
            df = df.drop(columns=["CLASS"])

        self.base_cols = [c for c in df.columns if c != "NSP"]

        # pairwise + monotonic
        df_aug = self.add_pairwise_ops(df)
        df_aug = self.add_monotonic(df_aug)

        # qbins
        self.qbin_edges_dict = {
            c: self.fit_qbins_edges(df_aug[c], q=5)
            for c in self.qbin_targets if c in df_aug.columns
        }
        for c, edges in self.qbin_edges_dict.items():
            if edges is not None:
                df_aug[f"{c}_qbin5"] = self.cut_with_edges(df_aug[c], edges)

        # outlier scorers
        num_cols = [c for c in df_aug.columns if c != "NSP"]
        self.lof, self.iso = self.fit_outlier_scorers(df_aug[num_cols].values)
        df_aug = self.transform_outlier_scores(df_aug, num_cols)

        # drop high correlation
        self.keep_cols = self.drop_high_corr_cols(df_aug.drop(columns=["NSP"]))
        return self

    def transform(self, df):
        df = df.copy()
        if "CLASS" in df.columns:
            df = df.drop(columns=["CLASS"])

        df = self.add_pairwise_ops(df)
        df = self.add_monotonic(df)

        for c, edges in self.qbin_edges_dict.items():
            if (edges is not None) and (c in df.columns):
                df[f"{c}_qbin5"] = self.cut_with_edges(df[c], edges)

        df = self.transform_outlier_scores(df, [c for c in df.columns if c != "NSP"])

        # align columns
        for col in self.keep_cols:
            if col not in df.columns:
                df[col] = 0
        df_out = df[self.keep_cols].copy()

        # ensure NSP column is kept if present in input
        if "NSP" in df.columns:
            df_out["NSP"] = df["NSP"].values
        return df_out

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)
