"""
model.py — FraudGenome Model Experiment

Temporal experiment: Train on T1+T2 (normal + known fraud), test on T3 (mutated fraud).
Two models: Baseline (behavioral-only features) vs FraudGenome (+ graph/community/sharing).
Primary metric: PR-AUC (appropriate for extreme class imbalance).

IMPORTANT: graph_community_fraud_ratio is excluded due to confirmed label leakage.
See implementation plan for the full trace.
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

# Columns that are identifiers, labels, or leaked — never used as features
EXCLUDED_COLUMNS = [
    "account_id",
    "phase",
    "is_fraud_account",      # target label
    "fraud_cluster_id",      # target label
    "graph_community_fraud_ratio",  # LEAKAGE: computed from is_fraud ground truth
]

# Baseline: traditional behavioral features (no graph/community/sharing structure)
BASELINE_FEATURES = [
    # Account genes
    "acct_tx_count", "acct_tx_velocity", "acct_active_days",
    "acct_amount_total", "acct_amount_avg", "acct_amount_std",
    "acct_age_days", "acct_kyc_unverified",
    # Device genes (individual behavior only)
    "dev_unique_count", "dev_tx_per_device", "dev_concentration",
    # Network genes (individual behavior only)
    "net_unique_ip_count", "net_tx_per_ip", "net_ip_concentration",
    # Transaction genes
    "tx_velocity", "tx_amount_mean", "tx_amount_max",
    "tx_amount_concentration", "tx_time_spread_hours",
    "tx_hour_mode", "tx_weekend_ratio",
    # Merchant genes (individual behavior only)
    "merch_unique_count", "merch_concentration",
    "merch_top_category", "merch_avg_age_days",
]

# Genome-only: relationship-sharing + graph/community structural features
GENOME_ONLY_FEATURES = [
    # Device sharing genes
    "dev_max_accts_per_device", "dev_shared_count", "dev_shared_ratio",
    # Network sharing genes
    "net_max_accts_per_ip", "net_shared_ip_count", "net_shared_ip_ratio",
    # Merchant sharing gene
    "merch_max_accts_per_merchant",
    # Graph structural genes
    "graph_degree", "graph_weighted_degree",
    "graph_connected_devices", "graph_connected_ips", "graph_connected_merchants",
    "graph_shared_devices", "graph_shared_ips",
    "graph_proj_degree", "graph_proj_weighted_degree",
    # Community topology genes (NO fraud_ratio — that's leaked)
    "graph_community_size", "graph_community_density",
    "graph_community_shared_device_ratio", "graph_community_shared_ip_ratio",
]

# FraudGenome = Baseline + Genome-only
FRAUDGENOME_FEATURES = BASELINE_FEATURES + GENOME_ONLY_FEATURES


# ---------------------------------------------------------------------------
# 1. Load features
# ---------------------------------------------------------------------------

def load_features(data_dir=None):
    """Load the normalized genome CSV."""
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")

    df = pd.read_csv(os.path.join(data_dir, "genome_full.csv"))
    return df, data_dir


# ---------------------------------------------------------------------------
# 2. Feature audit
# ---------------------------------------------------------------------------

def audit_features(df):
    """Print an explicit feature audit report, identifying leakage."""
    all_cols = list(df.columns)
    identifiers = ["account_id"]
    targets = ["is_fraud_account", "fraud_cluster_id"]
    metadata = ["phase"]
    leakage = ["graph_community_fraud_ratio"]

    baseline = BASELINE_FEATURES
    genome_only = GENOME_ONLY_FEATURES

    used_total = set(baseline + genome_only)
    remaining = [c for c in all_cols if c not in used_total
                 and c not in identifiers and c not in targets
                 and c not in metadata and c not in leakage]

    print("\n" + "=" * 70)
    print("  FEATURE AUDIT REPORT")
    print("=" * 70)
    print(f"\n  Total columns in genome_full.csv: {len(all_cols)}")
    print(f"\n  Identifier columns ({len(identifiers)}):")
    for c in identifiers:
        print(f"    - {c}")
    print(f"\n  Target/label columns ({len(targets)}):")
    for c in targets:
        print(f"    - {c}")
    print(f"\n  Metadata columns ({len(metadata)}):")
    for c in metadata:
        print(f"    - {c}")

    print(f"\n  ⚠️  EXCLUDED — Label Leakage ({len(leakage)}):")
    for c in leakage:
        print(f"    - {c}")
        print(f"      Trace: community.py computes fraud_ratio from is_fraud ground truth.")
        print(f"      community_stats → features.py → graph_community_fraud_ratio")
        print(f"      Using this feature = reading the answer sheet.")

    print(f"\n  Baseline candidate features ({len(baseline)}):")
    for c in baseline:
        present = "✓" if c in df.columns else "✗ MISSING"
        print(f"    - {c}  {present}")

    print(f"\n  Genome-only candidate features ({len(genome_only)}):")
    for c in genome_only:
        present = "✓" if c in df.columns else "✗ MISSING"
        print(f"    - {c}  {present}")

    print(f"\n  FraudGenome total features: {len(baseline)} + {len(genome_only)} = {len(baseline) + len(genome_only)}")

    if remaining:
        print(f"\n  Unaccounted columns ({len(remaining)}):")
        for c in remaining:
            print(f"    - {c}")
    else:
        print(f"\n  ✓ All columns accounted for.")

    print("=" * 70 + "\n")

    # Verify all feature columns actually exist
    missing = [c for c in FRAUDGENOME_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing from genome_full.csv: {missing}")


# ---------------------------------------------------------------------------
# 3. Temporal split
# ---------------------------------------------------------------------------

def prepare_temporal_split(df):
    """Split data into temporal train (T1+T2) and test (T3) sets.

    Returns train_df, test_df, and a T2-only reference DataFrame.
    """
    train_df = df[df["phase"].isin(["T1", "T2"])].copy().reset_index(drop=True)
    test_df = df[df["phase"] == "T3"].copy().reset_index(drop=True)
    t2_ref_df = df[df["phase"] == "T2"].copy().reset_index(drop=True)

    print("=" * 70)
    print("  TEMPORAL EXPERIMENT SPLIT")
    print("=" * 70)
    print(f"\n  TRAIN (T1 + T2):")
    print(f"    T1 rows: {len(df[df['phase'] == 'T1'])}")
    print(f"    T2 rows: {len(df[df['phase'] == 'T2'])}")
    print(f"    Total:   {len(train_df)}")
    print(f"    Fraud:   {train_df['is_fraud_account'].sum()} "
          f"({100 * train_df['is_fraud_account'].mean():.2f}%)")

    print(f"\n  TEST (T3):")
    print(f"    Total:   {len(test_df)}")
    print(f"    Fraud:   {test_df['is_fraud_account'].sum()} "
          f"({100 * test_df['is_fraud_account'].mean():.2f}%)")

    print(f"\n  REFERENCE (T2 only, for degradation measurement):")
    print(f"    Total:   {len(t2_ref_df)}")
    print(f"    Fraud:   {t2_ref_df['is_fraud_account'].sum()} "
          f"({100 * t2_ref_df['is_fraud_account'].mean():.2f}%)")
    print("=" * 70 + "\n")

    return train_df, test_df, t2_ref_df


# ---------------------------------------------------------------------------
# 4. Model training
# ---------------------------------------------------------------------------

def train_model(X_train, y_train, model_name, seed=42):
    """Train an XGBoost classifier with appropriate class imbalance handling.

    Uses scale_pos_weight = n_negatives / n_positives to handle the extreme
    imbalance (~0.25% fraud in training set).
    """
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / max(n_pos, 1)

    print(f"  Training {model_name}...")
    print(f"    Features:         {X_train.shape[1]}")
    print(f"    Training rows:    {X_train.shape[0]}")
    print(f"    Positive class:   {n_pos}")
    print(f"    Negative class:   {n_neg}")
    print(f"    scale_pos_weight: {scale_pos_weight:.1f}")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(X_train, y_train, verbose=False)
    print(f"    ✓ {model_name} trained.\n")

    return model


# ---------------------------------------------------------------------------
# 5. Model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, label=""):
    """Evaluate a trained model on a test set. Returns metrics dict."""
    y_prob = model.predict_proba(X_test)[:, 1]

    # Use 0.5 threshold for hard predictions
    y_pred = (y_prob >= 0.5).astype(int)

    # Handle edge case: if no positive predictions or no positives in y_test
    n_pos = int(y_test.sum())
    if n_pos == 0:
        print(f"  [{label}] No positive samples in evaluation set.")
        return {"pr_auc": 0.0, "roc_auc": 0.0, "precision": 0.0,
                "recall": 0.0, "f1": 0.0, "y_prob": y_prob, "y_pred": y_pred}

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "pr_auc": float(round(pr_auc, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "confusion_matrix": cm.tolist(),
        "n_positive": int(n_pos),
        "n_total": len(y_test),
        "y_prob": y_prob,
        "y_pred": y_pred,
    }

    return metrics


def print_metrics(metrics, label):
    """Print evaluation metrics for a single model/phase combination."""
    print(f"  [{label}]")
    print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
    print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"    F1:        {metrics['f1']:.4f}")
    if "confusion_matrix" in metrics:
        cm = metrics["confusion_matrix"]
        print(f"    Confusion Matrix:")
        print(f"      TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
        print(f"      FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")
    print()


# ---------------------------------------------------------------------------
# 6. Compare models
# ---------------------------------------------------------------------------

def compare_models(
    baseline_t3, genome_t3,
    baseline_t2, genome_t2,
):
    """Print model comparison table and compute degradation / improvement."""
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON")
    print("=" * 70)

    # T3 comparison (main experiment)
    header = f"  {'Metric':<18} {'Baseline':>12} {'FraudGenome':>14} {'Δ Improvement':>16}"
    print(f"\n  T3 (Mutated Fraud — Primary Evaluation)")
    print(f"  {'-'*62}")
    print(header)
    print(f"  {'-'*62}")
    for metric_key, metric_name in [
        ("pr_auc",    "PR-AUC"),
        ("roc_auc",   "ROC-AUC"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1"),
    ]:
        b_val = baseline_t3[metric_key]
        g_val = genome_t3[metric_key]
        delta = g_val - b_val
        sign = "+" if delta >= 0 else ""
        print(f"  {metric_name:<18} {b_val:>12.4f} {g_val:>14.4f} {sign}{delta:>15.4f}")
    print()

    # T2 reference
    print(f"  T2 (Known Fraud — Reference Evaluation)")
    print(f"  {'-'*62}")
    print(header)
    print(f"  {'-'*62}")
    for metric_key, metric_name in [
        ("pr_auc",    "PR-AUC"),
        ("roc_auc",   "ROC-AUC"),
        ("precision", "Precision"),
        ("recall",    "Recall"),
        ("f1",        "F1"),
    ]:
        b_val = baseline_t2[metric_key]
        g_val = genome_t2[metric_key]
        delta = g_val - b_val
        sign = "+" if delta >= 0 else ""
        print(f"  {metric_name:<18} {b_val:>12.4f} {g_val:>14.4f} {sign}{delta:>15.4f}")
    print()

    # Performance degradation: T2 → T3
    print(f"  PERFORMANCE DEGRADATION (T2 Known → T3 Mutated)")
    print(f"  {'-'*62}")
    print(f"  {'Metric':<18} {'Baseline Δ':>12} {'FraudGenome Δ':>14}")
    print(f"  {'-'*62}")
    for metric_key, metric_name in [
        ("pr_auc",  "PR-AUC"),
        ("roc_auc", "ROC-AUC"),
        ("f1",      "F1"),
    ]:
        b_deg = baseline_t3[metric_key] - baseline_t2[metric_key]
        g_deg = genome_t3[metric_key] - genome_t2[metric_key]
        print(f"  {metric_name:<18} {b_deg:>+12.4f} {g_deg:>+14.4f}")
    print()

    # Net improvement on T3
    t3_improvement_prauc = genome_t3["pr_auc"] - baseline_t3["pr_auc"]
    t3_improvement_f1 = genome_t3["f1"] - baseline_t3["f1"]
    print(f"  FraudGenome improvement on T3 (mutated fraud):")
    print(f"    PR-AUC: {t3_improvement_prauc:+.4f}")
    print(f"    F1:     {t3_improvement_f1:+.4f}")
    print("=" * 70 + "\n")

    return {
        "t3_improvement_prauc": float(round(t3_improvement_prauc, 4)),
        "t3_improvement_f1": float(round(t3_improvement_f1, 4)),
        "baseline_degradation_prauc": float(round(baseline_t3["pr_auc"] - baseline_t2["pr_auc"], 4)),
        "genome_degradation_prauc": float(round(genome_t3["pr_auc"] - genome_t2["pr_auc"], 4)),
    }


# ---------------------------------------------------------------------------
# 7. Save results
# ---------------------------------------------------------------------------

def save_predictions(model, df, features, model_name, data_dir):
    """Save prediction CSV with account_id, phase, scenario, is_fraud, probability, label."""
    X = df[features].values
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    out = pd.DataFrame({
        "account_id": df["account_id"].values,
        "phase": df["phase"].values,
        "is_fraud_account": df["is_fraud_account"].values,
        "predicted_probability": np.round(y_prob, 6),
        "predicted_label": y_pred,
    })

    path = os.path.join(data_dir, f"{model_name}_predictions.csv")
    out.to_csv(path, index=False)
    print(f"  Saved {path}")
    return out


def save_results(
    audit_info, train_info, metrics_dict, comparison, data_dir
):
    """Save model_results.json with full experiment metadata."""
    # Strip non-serializable keys from metrics
    clean_metrics = {}
    for key, val in metrics_dict.items():
        if isinstance(val, dict):
            clean_metrics[key] = {
                k: v for k, v in val.items()
                if k not in ("y_prob", "y_pred")
            }
        else:
            clean_metrics[key] = val

    results = {
        "feature_audit": audit_info,
        "train_test_split": train_info,
        "metrics": clean_metrics,
        "comparison": comparison,
    }

    path = os.path.join(data_dir, "model_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved {path}")


def save_models(baseline_model, genome_model, data_dir):
    """Save trained model objects as pickle files."""
    model_dir = os.path.join(data_dir, "models")
    os.makedirs(model_dir, exist_ok=True)

    baseline_path = os.path.join(model_dir, "baseline_xgb.pkl")
    genome_path = os.path.join(model_dir, "fraudgenome_xgb.pkl")

    with open(baseline_path, "wb") as f:
        pickle.dump(baseline_model, f)
    with open(genome_path, "wb") as f:
        pickle.dump(genome_model, f)

    print(f"  Saved {baseline_path}")
    print(f"  Saved {genome_path}")


# ---------------------------------------------------------------------------
# 8. Main pipeline
# ---------------------------------------------------------------------------

def main():
    df, data_dir = load_features()

    # ---- Step 1: Feature audit ----
    audit_features(df)

    # ---- Step 2: Temporal split ----
    train_df, test_df, t2_ref_df = prepare_temporal_split(df)

    # ---- Step 3: Prepare feature matrices ----
    X_train_baseline = train_df[BASELINE_FEATURES].values
    X_train_genome   = train_df[FRAUDGENOME_FEATURES].values
    y_train          = train_df["is_fraud_account"].astype(int).values

    X_test_baseline  = test_df[BASELINE_FEATURES].values
    X_test_genome    = test_df[FRAUDGENOME_FEATURES].values
    y_test           = test_df["is_fraud_account"].astype(int).values

    X_t2_baseline    = t2_ref_df[BASELINE_FEATURES].values
    X_t2_genome      = t2_ref_df[FRAUDGENOME_FEATURES].values
    y_t2             = t2_ref_df["is_fraud_account"].astype(int).values

    # ---- Step 4: Train models ----
    print("=" * 70)
    print("  MODEL TRAINING")
    print("=" * 70 + "\n")

    baseline_model = train_model(X_train_baseline, y_train, "Baseline (XGBoost)")
    genome_model   = train_model(X_train_genome, y_train, "FraudGenome (XGBoost)")

    # ---- Step 5: Evaluate on T3 (primary) ----
    print("=" * 70)
    print("  EVALUATION — T3 (Mutated Fraud)")
    print("=" * 70 + "\n")

    baseline_t3 = evaluate_model(baseline_model, X_test_baseline, y_test, "Baseline / T3")
    print_metrics(baseline_t3, "Baseline / T3")

    genome_t3 = evaluate_model(genome_model, X_test_genome, y_test, "FraudGenome / T3")
    print_metrics(genome_t3, "FraudGenome / T3")

    # ---- Step 6: Evaluate on T2 (reference — known fraud) ----
    # Note: T2 is IN the training set, so these metrics represent in-sample
    # performance on known fraud. This is intentional: we want to measure
    # how well the model captures the known pattern as a reference point
    # for computing degradation when facing the mutated T3 pattern.
    print("=" * 70)
    print("  EVALUATION — T2 (Known Fraud — Reference, in-sample)")
    print("=" * 70 + "\n")

    baseline_t2 = evaluate_model(baseline_model, X_t2_baseline, y_t2, "Baseline / T2")
    print_metrics(baseline_t2, "Baseline / T2 (reference)")

    genome_t2 = evaluate_model(genome_model, X_t2_genome, y_t2, "FraudGenome / T2")
    print_metrics(genome_t2, "FraudGenome / T2 (reference)")

    # ---- Step 7: Compare ----
    comparison = compare_models(baseline_t3, genome_t3, baseline_t2, genome_t2)

    # ---- Step 8: Save everything ----
    print("=" * 70)
    print("  SAVING ARTIFACTS")
    print("=" * 70 + "\n")

    audit_info = {
        "total_columns": len(df.columns),
        "excluded_columns": EXCLUDED_COLUMNS,
        "leakage_features": ["graph_community_fraud_ratio"],
        "leakage_explanation": (
            "graph_community_fraud_ratio is computed from is_fraud ground truth "
            "labels in community.py line 190. It encodes the fraction of accounts "
            "in a Louvain community that are labeled fraudulent."
        ),
        "baseline_features": BASELINE_FEATURES,
        "baseline_feature_count": len(BASELINE_FEATURES),
        "genome_only_features": GENOME_ONLY_FEATURES,
        "genome_only_feature_count": len(GENOME_ONLY_FEATURES),
        "fraudgenome_total_features": len(FRAUDGENOME_FEATURES),
    }

    train_info = {
        "train_phases": ["T1", "T2"],
        "test_phase": "T3",
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_fraud_count": int(train_df["is_fraud_account"].sum()),
        "train_fraud_pct": round(100 * train_df["is_fraud_account"].mean(), 4),
        "test_fraud_count": int(test_df["is_fraud_account"].sum()),
        "test_fraud_pct": round(100 * test_df["is_fraud_account"].mean(), 4),
    }

    # Strip numpy arrays before saving to JSON
    metrics_dict = {
        "baseline_t3": baseline_t3,
        "genome_t3": genome_t3,
        "baseline_t2_reference": baseline_t2,
        "genome_t2_reference": genome_t2,
    }

    save_results(audit_info, train_info, metrics_dict, comparison, data_dir)

    # Save prediction CSVs (on full genome for all phases)
    save_predictions(baseline_model, df, BASELINE_FEATURES, "baseline", data_dir)
    save_predictions(genome_model, df, FRAUDGENOME_FEATURES, "genome", data_dir)

    # Save trained models
    save_models(baseline_model, genome_model, data_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
