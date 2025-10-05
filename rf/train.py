# train.py
# ------------------------------------------------------------
# Model Training Script (Random Forest + Feature Engineering)
# Reads data/train.csv and data/test.csv, applies FE,
# trains a Random Forest, evaluates, and saves model weights.
# ------------------------------------------------------------

import os
import numpy as np
import pandas as pd
import joblib, torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from features_fe import FeatureEngineer

RANDOM_STATE = 3407

class RF_FE_Model:
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.rf = RandomForestClassifier(
            n_estimators=1200,
            max_depth=16,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="sqrt",
            class_weight=None,
            bootstrap=True,
            oob_score=True,
            random_state=random_state,
            n_jobs=-1,
        )
        self.fe = FeatureEngineer(random_state=random_state)

    def train(self, df_tr):
        print("🧩 Starting Feature Engineering...")
        tr_final = self.fe.fit_transform(df_tr)
        Xtr = tr_final.drop(columns=["NSP"]).values
        ytr = tr_final["NSP"].values

        print("🌲 Training Random Forest...")
        self.rf.fit(Xtr, ytr)
        print("✅ Training complete!")

    def evaluate(self, df_te):
        print("📊 Evaluating model...")
        te_final = self.fe.transform(df_te)
        Xte = te_final.drop(columns=["NSP"]).values
        yte = te_final["NSP"].values

        pred = self.rf.predict(Xte)
        proba = self.rf.predict_proba(Xte)

        print("\n=== Evaluation on Test Set ===")
        print("Accuracy:", accuracy_score(yte, pred))
        print("Balanced Acc:", balanced_accuracy_score(yte, pred))
        print("Macro-F1:", f1_score(yte, pred, average="macro"))
        print("\nClassification report:\n", classification_report(yte, pred, digits=4))
        print("Confusion matrix:\n", confusion_matrix(yte, pred))
        print("ROC AUC (OVO, macro):", roc_auc_score(yte, proba, multi_class="ovo", average="macro"))

    def save(self, path_dir="models"):
        os.makedirs(path_dir, exist_ok=True)
        joblib.dump({"model": self.rf, "fe": self.fe}, f"{path_dir}/rf_fe.joblib")
        torch.save({"model": self.rf, "fe": self.fe}, f"{path_dir}/rf_fe.pt")
        print(f"✅ Saved trained model to {path_dir}/rf_fe.pt and rf_fe.joblib")


# ------------------------------------------------------------
# MAIN EXECUTION BLOCK
# ------------------------------------------------------------
if __name__ == "__main__":
    print("📥 Loading training/test data...")
    df_train = pd.read_csv("data/train.csv")
    df_test = pd.read_csv("data/test.csv")

    trainer = RF_FE_Model(random_state=RANDOM_STATE)
    trainer.train(df_train)
    trainer.evaluate(df_test)
    trainer.save()
