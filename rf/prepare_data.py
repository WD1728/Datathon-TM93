# -*- coding: utf-8 -*-
"""
prepare_data.py
--------------
Fetches the official UCI Cardiotocography dataset (id=193),
drops CLASS column, performs cleaning + scaling + SMOTE balancing,
and saves train/test splits to data/train.csv and data/test.csv.
"""

from pathlib import Path
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 3407
TEST_SIZE = 0.2
USE_SMOTE = True

def preprocess_ctg(
    df: pd.DataFrame,
    target_col: str = "NSP",
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    use_smote: bool = USE_SMOTE
):
    # Drop leakage column if exists
    if "CLASS" in df.columns and target_col != "CLASS":
        df = df.drop(columns=["CLASS"])
    
    # Separate features and target
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols].to_numpy()
    y_raw = df[target_col].to_numpy().reshape(-1)

    # Map target NSP from {1,2,3} → {0,1,2}
    if np.array_equal(np.unique(y_raw), np.array([1, 2, 3])):
        y = y_raw.astype(int) - 1
    else:
        _, y = np.unique(y_raw, return_inverse=True)

    # Split train/test (stratified)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Scale (fit on train, apply on test)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    # SMOTE (only training)
    if use_smote:
        sm = SMOTE(random_state=random_state)
        X_tr, y_tr = sm.fit_resample(X_tr, y_tr)

    # Combine into DataFrames for saving
    df_tr = pd.DataFrame(X_tr, columns=feature_cols)
    df_tr["NSP"] = y_tr
    df_te = pd.DataFrame(X_te, columns=feature_cols)
    df_te["NSP"] = y_te

    return df_tr, df_te


def main():
    print("Fetching UCI Cardiotocography dataset (id=193)...")
    cardiotocography = fetch_ucirepo(id=193)
    X = cardiotocography.data.features
    y = cardiotocography.data.targets

    df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
    print("✅ Loaded dataset with shape:", df.shape)

    # Run preprocessing
    df_train, df_test = preprocess_ctg(df)

    # Create output folder
    Path("data").mkdir(exist_ok=True)

    # Save CSVs
    df_train.to_csv("data/train.csv", index=False)
    df_test.to_csv("data/test.csv", index=False)

    print("✅ Saved data/train.csv and data/test.csv")
    print("Train shape:", df_train.shape, "| Test shape:", df_test.shape)
    print("Train label counts:", np.bincount(df_train["NSP"].astype(int)))
    print("Test label counts:", np.bincount(df_test["NSP"].astype(int)))


if __name__ == "__main__":
    main()
