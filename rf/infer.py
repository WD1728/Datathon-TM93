# infer.py
# ------------------------------------------------------------
# Inference script for Random Forest + Feature Engineering.
# Supports both .joblib (preferred) and .pt (PyTorch ≥2.6 safe load).
# ------------------------------------------------------------

import argparse
import os
import pandas as pd
import numpy as np
import torch, joblib

from features_fe import FeatureEngineer   


# === Default paths ===
DEFAULT_MODEL_JL   = "models/rf_fe.joblib"
DEFAULT_MODEL_PT   = "models/rf_fe.pt"
DEFAULT_INPUT_CSV  = "data/test.csv"
DEFAULT_OUTPUT_CSV = "predictions.csv"

# ------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using trained RF+FE model")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL_JL,
                        help="Path to model (.joblib or .pt)")
    parser.add_argument("--input", type=str, default=DEFAULT_INPUT_CSV,
                        help="Input CSV file path")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT_CSV,
                        help="Output CSV file path")
    return parser.parse_args()

# ------------------------------------------------------------
# Safe model loader (handles PyTorch 2.6 security defaults)
# ------------------------------------------------------------
def load_model(model_path: str):
    if model_path.endswith(".joblib"):
        return joblib.load(model_path)

    if model_path.endswith(".pt"):
        # Explicitly allow sklearn + FE classes when unpickling
        from sklearn.ensemble import RandomForestClassifier
        try:
            from fe import FeatureEngineer
            allow = [RandomForestClassifier, FeatureEngineer]
        except Exception:
            allow = [RandomForestClassifier]

        with torch.serialization.safe_globals(allow):
            return torch.load(model_path, map_location="cpu", weights_only=False)

    raise ValueError("Unsupported model file. Use .joblib or .pt")

# ------------------------------------------------------------
def drop_leakage_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove any leakage columns such as 'CLASS'."""
    return df.drop(columns=[c for c in df.columns if c.upper() == "CLASS"], errors="ignore")

# ------------------------------------------------------------
def main():
    args = parse_args()

    # 1️⃣ Load model
    if os.path.exists(args.model):
        blob = load_model(args.model)
    elif os.path.exists(DEFAULT_MODEL_JL):
        print(f"⚠️ {args.model} not found, using {DEFAULT_MODEL_JL} instead.")
        blob = load_model(DEFAULT_MODEL_JL)
    elif os.path.exists(DEFAULT_MODEL_PT):
        print(f"⚠️ Falling back to {DEFAULT_MODEL_PT}.")
        blob = load_model(DEFAULT_MODEL_PT)
    else:
        raise FileNotFoundError(f"Model not found at {args.model}")

    # 2️⃣ Extract components
    model, fe = None, None
    if isinstance(blob, dict):
        model = blob.get("model", blob)
        fe    = blob.get("fe", None)
    else:
        model = getattr(blob, "rf", blob)
        fe    = getattr(blob, "fe", None)

    if model is None:
        raise RuntimeError("Loaded model object is invalid or missing.")

    # 3️⃣ Load data
    df_raw = pd.read_csv(args.input)
    df_raw = drop_leakage_columns(df_raw)
    if "NSP" not in df_raw.columns:
        df_raw["NSP"] = -1

    # 4️⃣ Apply same FE
    if fe is not None:
        df_processed = fe.transform(df_raw.copy())
        X = df_processed.drop(columns=["NSP"], errors="ignore").values
    else:
        X = df_raw.drop(columns=["NSP"], errors="ignore").values

    # 5️⃣ Predict
    print("🔍 Running inference...")
    preds = model.predict(X)
    try:
        probas = model.predict_proba(X)
    except Exception:
        probas = None

    # 6️⃣ Save results
    out = df_raw.drop(columns=["NSP"], errors="ignore").copy()
    out["pred"] = preds
    if probas is not None:
        for i in range(probas.shape[1]):
            out[f"proba_{i}"] = probas[:, i]

    out.to_csv(args.output, index=False)
    print(f"✅ Predictions saved to {args.output}")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
