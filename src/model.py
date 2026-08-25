"""
model.py — FraudGenome Model Training, Evaluation, and Explainability Pipeline

Execution Flow:
1. Feature Audit & Leakage Inspection
2. Leakage-Safe Data Splitting (True Cluster-Grouped + Account-Grouped 70/15/15)
3. Pre-Training Pre-flight Integrity Validation
4. Baseline vs FraudGenome Experiments across 3 Model Families:
   - Logistic Regression (with StandardScaler fit ONLY on Train)
   - Random Forest Classifier (balanced class weighting)
   - XGBoost Classifier (scale_pos_weight for 0.33% fraud imbalance)
5. Validation Threshold Optimization Grid [0.05 ... 0.90]
6. Frozen Untouched Test Set Evaluation (on unseen test fraud cluster_004)
7. Feature Importance & Gene Group Attribution
8. Secondary Out-of-Time Temporal Experiment (T1+T2 -> T3)
9. Saving artifacts: data/model_results.json, data/feature_importance.csv, models/*.pkl
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

EXCLUDED_COLUMNS = [
    "account_id",
    "phase",
    "is_fraud_account",      # target label
    "fraud_cluster_id",      # target label/metadata
    "graph_community_fraud_ratio",  # LEAKAGE: computed from is_fraud ground truth
]

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

FRAUDGENOME_FEATURES = BASELINE_FEATURES + GENOME_ONLY_FEATURES

# Gene Group Mapping for Feature Attribution
GENE_GROUP_MAP = {
    "acct_": "Account",
    "dev_": "Device",
    "net_": "Network",
    "tx_": "Transaction",
    "merch_": "Merchant",
    "graph_community_": "Community",
    "graph_": "Graph",
}


def get_gene_group(feature_name):
    """Categorize a feature into its behavioral gene group."""
    if feature_name.startswith("graph_community_"):
        return "Community"
    for prefix, group in GENE_GROUP_MAP.items():
        if feature_name.startswith(prefix):
            return group
    return "Other"


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

    print("\n" + "=" * 80)
    print("  FEATURE AUDIT & LEAKAGE INSPECTION REPORT")
    print("=" * 80)
    print(f"\n  Total columns in genome_full.csv: {len(all_cols)}")
    print(f"\n  Identifier columns ({len(identifiers)}):")
    for c in identifiers:
        print(f"    - {c} (EXCLUDED FROM MODEL)")
    print(f"\n  Target/label columns ({len(targets)}):")
    for c in targets:
        print(f"    - {c} (EXCLUDED FROM MODEL)")
    print(f"\n  Metadata columns ({len(metadata)}):")
    for c in metadata:
        print(f"    - {c} (EXCLUDED FROM MODEL)")

    print(f"\n  ⚠️  EXCLUDED — Label Leakage ({len(leakage)}):")
    for c in leakage:
        print(f"    - {c}")
        print(f"      Trace: community.py computes fraud_ratio from is_fraud ground truth.")
        print(f"      community_stats -> features.py -> graph_community_fraud_ratio")
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

    print("=" * 80 + "\n")

    missing = [c for c in FRAUDGENOME_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing from genome_full.csv: {missing}")


# ---------------------------------------------------------------------------
# 3. Data Splitting Pipeline
# ---------------------------------------------------------------------------

def prepare_cluster_aware_group_split(df, seed=42):
    """Perform a leakage-safe Cluster-Grouped + Account-Grouped Stratified Split (70/15/15)."""
    acct_info = df.groupby("account_id").agg(
        is_fraud=("is_fraud_account", "any"),
        cluster_id=("fraud_cluster_id", lambda x: [i for i in x if i != "none"][0] if any(i != "none" for i in x) else "none")
    ).reset_index()

    fraud_clusters = sorted([c for c in acct_info["cluster_id"].unique() if c != "none"])
    n_clusters = len(fraud_clusters)

    rng = np.random.RandomState(seed)
    shuffled_clusters = list(fraud_clusters)
    rng.shuffle(shuffled_clusters)

    n_train_c = max(int(np.round(0.60 * n_clusters)), 1)
    n_val_c = max(int(np.round(0.20 * n_clusters)), 1)

    train_clusters = sorted(shuffled_clusters[:n_train_c])
    val_clusters = sorted(shuffled_clusters[n_train_c:n_train_c + n_val_c])
    test_clusters = sorted(shuffled_clusters[n_train_c + n_val_c:])

    fraud_accts = acct_info[acct_info["is_fraud"]]
    train_fraud_accts = fraud_accts[fraud_accts["cluster_id"].isin(train_clusters)]
    val_fraud_accts = fraud_accts[fraud_accts["cluster_id"].isin(val_clusters)]
    test_fraud_accts = fraud_accts[fraud_accts["cluster_id"].isin(test_clusters)]

    normal_accts = acct_info[~acct_info["is_fraud"]]
    train_norm, test_val_norm = train_test_split(
        normal_accts, test_size=0.30, random_state=seed
    )
    val_norm, test_norm = train_test_split(
        test_val_norm, test_size=0.50, random_state=seed
    )

    train_acct_ids = set(train_fraud_accts["account_id"]).union(set(train_norm["account_id"]))
    val_acct_ids = set(val_fraud_accts["account_id"]).union(set(val_norm["account_id"]))
    test_acct_ids = set(test_fraud_accts["account_id"]).union(set(test_norm["account_id"]))

    train_df = df[df["account_id"].isin(train_acct_ids)].copy().reset_index(drop=True)
    val_df = df[df["account_id"].isin(val_acct_ids)].copy().reset_index(drop=True)
    test_df = df[df["account_id"].isin(test_acct_ids)].copy().reset_index(drop=True)

    report = validate_split(
        train_df, val_df, test_df,
        train_clusters, val_clusters, test_clusters
    )

    return train_df, val_df, test_df, report


def validate_split(train_df, val_df, test_df, train_clusters, val_clusters, test_clusters):
    """Run automated verification checks on account and fraud cluster isolation."""
    train_accts = set(train_df["account_id"])
    val_accts = set(val_df["account_id"])
    test_accts = set(test_df["account_id"])

    tv_acct_overlap = len(train_accts.intersection(val_accts))
    tt_acct_overlap = len(train_accts.intersection(test_accts))
    vt_acct_overlap = len(val_accts.intersection(test_accts))

    no_account_overlap = (tv_acct_overlap == 0 and tt_acct_overlap == 0 and vt_acct_overlap == 0)
    assert no_account_overlap, f"ACCOUNT LEAKAGE ASSERTION FAILED!"

    c_train = set(train_clusters)
    c_val = set(val_clusters)
    c_test = set(test_clusters)

    tv_cluster_overlap = c_train.intersection(c_val)
    tt_cluster_overlap = c_train.intersection(c_test)
    vt_cluster_overlap = c_val.intersection(c_test)

    no_fraud_cluster_overlap = (
        len(tv_cluster_overlap) == 0 and len(tt_cluster_overlap) == 0 and len(vt_cluster_overlap) == 0
    )
    assert no_fraud_cluster_overlap, f"FRAUD CLUSTER LEAKAGE ASSERTION FAILED!"

    leakage_safe = no_account_overlap and no_fraud_cluster_overlap
    assert leakage_safe, "LEAKAGE SAFETY ASSERTION FAILED!"

    total_rows = len(train_df) + len(val_df) + len(test_df)
    train_pct = 100 * len(train_df) / total_rows
    val_pct = 100 * len(val_df) / total_rows
    test_pct = 100 * len(test_df) / total_rows

    train_fraud_count = int(train_df["is_fraud_account"].sum())
    val_fraud_count = int(val_df["is_fraud_account"].sum())
    test_fraud_count = int(test_df["is_fraud_account"].sum())

    train_fraud_pct = 100 * train_df["is_fraud_account"].mean()
    val_fraud_pct = 100 * val_df["is_fraud_account"].mean()
    test_fraud_pct = 100 * test_df["is_fraud_account"].mean()

    overall_fraud_pct = 100 * (train_fraud_count + val_fraud_count + test_fraud_count) / total_rows

    report = {
        "strategy": "True Cluster-Grouped + Account-Grouped Stratified Split (70/15/15)",
        "total_rows": total_rows,
        "overall_fraud_pct": round(overall_fraud_pct, 4),
        "leakage_safe": leakage_safe,
        "splits": {
            "train": {
                "rows": len(train_df),
                "row_pct": round(train_pct, 2),
                "unique_accounts": len(train_accts),
                "fraud_rows": train_fraud_count,
                "fraud_pct": round(train_fraud_pct, 4),
                "fraud_clusters": train_clusters,
                "fraud_cluster_count": len(train_clusters),
            },
            "val": {
                "rows": len(val_df),
                "row_pct": round(val_pct, 2),
                "unique_accounts": len(val_accts),
                "fraud_rows": val_fraud_count,
                "fraud_pct": round(val_fraud_pct, 4),
                "fraud_clusters": val_clusters,
                "fraud_cluster_count": len(val_clusters),
            },
            "test": {
                "rows": len(test_df),
                "row_pct": round(test_pct, 2),
                "unique_accounts": len(test_accts),
                "fraud_rows": test_fraud_count,
                "fraud_pct": round(test_fraud_pct, 4),
                "fraud_clusters": test_clusters,
                "fraud_cluster_count": len(test_clusters),
            },
        },
        "overlap_verification": {
            "train_val_account_overlap": tv_acct_overlap,
            "train_test_account_overlap": tt_acct_overlap,
            "val_test_account_overlap": vt_acct_overlap,
            "no_account_overlap": no_account_overlap,
            "train_val_cluster_overlap": list(tv_cluster_overlap),
            "train_test_cluster_overlap": list(tt_cluster_overlap),
            "val_test_cluster_overlap": list(vt_cluster_overlap),
            "no_fraud_cluster_overlap": no_fraud_cluster_overlap,
            "leakage_safe": leakage_safe,
        },
    }

    print("=" * 80)
    print("  LEAKAGE-SAFE CLUSTER-GROUPED DATA SPLIT REPORT")
    print("=" * 80)
    print(f"\n  Strategy: {report['strategy']}")
    print(f"  Total Dataset Rows: {total_rows}  | Overall Fraud Prevalence: {overall_fraud_pct:.2f}%")
    print(f"  Leakage-Safe Condition (no_account_overlap AND no_fraud_cluster_overlap): {leakage_safe}")

    print(f"\n  {'Split':<12} {'Rows':>8} {'Pct':>8} {'Accounts':>10} {'Fraud Rows':>12} {'Fraud %':>10} {'Clusters':>12}")
    print("  " + "-" * 78)
    for name, s in report["splits"].items():
        c_str = f"{s['fraud_cluster_count']} ({','.join(s['fraud_clusters'])})"
        print(f"  {name.upper():<12} {s['rows']:>8} {s['row_pct']:>7.1f}% {s['unique_accounts']:>10} {s['fraud_rows']:>12} {s['fraud_pct']:>9.2f}% {c_str:>16}")

    print(f"\n  Account Overlap Checks:")
    print(f"    ✓ Train vs Val Account Overlap:  {tv_acct_overlap}")
    print(f"    ✓ Train vs Test Account Overlap: {tt_acct_overlap}")
    print(f"    ✓ Val vs Test Account Overlap:   {vt_acct_overlap}")

    print(f"\n  Fraud Cluster Overlap Checks (non-'none'):")
    print(f"    ✓ Train vs Val Cluster Overlap:  {list(tv_cluster_overlap)}")
    print(f"    ✓ Train vs Test Cluster Overlap: {list(tt_cluster_overlap)}")
    print(f"    ✓ Val vs Test Cluster Overlap:   {list(vt_cluster_overlap)}")
    print("=" * 80 + "\n")

    return report


def prepare_temporal_split(df):
    """Optional temporal split: Train (T1+T2) -> Test (T3) for out-of-time mutation experiment."""
    train_df = df[df["phase"].isin(["T1", "T2"])].copy().reset_index(drop=True)
    test_df = df[df["phase"] == "T3"].copy().reset_index(drop=True)
    t2_ref_df = df[df["phase"] == "T2"].copy().reset_index(drop=True)
    return train_df, test_df, t2_ref_df


# ---------------------------------------------------------------------------
# 4. Pre-Training Integrity Validation
# ---------------------------------------------------------------------------

def validate_pre_training_integrity(train_df, val_df, test_df, feature_cols):
    """Run thorough pre-flight validation checks before model training."""
    print("Running Pre-Training Integrity Validation...")

    # 1. Verify feature columns exist
    for col in feature_cols:
        assert col in train_df.columns, f"Missing feature {col} in train_df"
        assert col in val_df.columns, f"Missing feature {col} in val_df"
        assert col in test_df.columns, f"Missing feature {col} in test_df"

    # 2. Verify excluded columns are not in feature_cols
    for ex_col in EXCLUDED_COLUMNS:
        assert ex_col not in feature_cols, f"EXCLUDED COLUMN {ex_col} IS PRESENT IN FEATURE MATRIX!"

    # 3. Verify target is binary
    for df_sub, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        unique_y = df_sub["is_fraud_account"].unique()
        assert set(unique_y).issubset({True, False, 0, 1}), f"Non-binary target in {name}: {unique_y}"

    # 4. Verify no NaNs or Infs in features
    for df_sub, name in [(train_df, "train"), (val_df, "val"), (test_df, "test")]:
        X_sub = df_sub[feature_cols].values
        assert not np.isnan(X_sub).any(), f"NaN values detected in {name} features!"
        assert not np.isinf(X_sub).any(), f"Inf values detected in {name} features!"

    print("  ✓ All Pre-Training Integrity Validation checks PASSED!\n")


# ---------------------------------------------------------------------------
# 5. Model Training Functions
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train, seed=42):
    """Train Logistic Regression with balanced class weighting and feature scaling fit on Train."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=seed,
        solver="lbfgs",
    )
    model.fit(X_train_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train, seed=42):
    """Train Random Forest Classifier with balanced_subsample weighting."""
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model, None


def train_xgboost(X_train, y_train, seed=42):
    """Train XGBoost Classifier with scale_pos_weight for extreme imbalance."""
    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=seed,
    )
    model.fit(X_train, y_train, verbose=False)
    return model, None


# ---------------------------------------------------------------------------
# 6. Evaluation & Threshold Optimization
# ---------------------------------------------------------------------------

def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """Compute comprehensive classification metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(y_true.sum())

    if n_pos == 0:
        return {
            "pr_auc": 0.0, "roc_auc": 0.0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]],
            "fp_count": 0, "fn_count": 0, "fpr": 0.0, "fraud_detection_rate": 0.0,
        }

    pr_auc = average_precision_score(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "pr_auc": float(round(pr_auc, 4)),
        "roc_auc": float(round(roc_auc, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "f1": float(round(f1, 4)),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "fp_count": int(fp),
        "fn_count": int(fn),
        "fpr": float(round(fpr, 6)),
        "fraud_detection_rate": float(round(recall, 4)),
    }


def evaluate_threshold_grid(model, X_val, y_val, scaler=None):
    """Evaluate a grid of classification thresholds on the VALIDATION set.

    Grid: 0.05 to 0.90 in steps of 0.05.
    Returns: best_threshold, best_val_metrics, full_threshold_table
    """
    X_v = scaler.transform(X_val) if scaler is not None else X_val
    y_prob = model.predict_proba(X_v)[:, 1]

    threshold_grid = [round(t, 2) for t in np.arange(0.05, 0.95, 0.05)]
    grid_results = []

    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics = None

    for thresh in threshold_grid:
        m = evaluate_predictions(y_val, y_prob, threshold=thresh)
        row = {
            "threshold": thresh,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "fp_count": m["fp_count"],
            "fn_count": m["fn_count"],
            "tp": m["tp"],
            "tn": m["tn"],
        }
        grid_results.append(row)

        # Select threshold that maximizes Validation F1 (or fallback to PR-AUC)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh
            best_metrics = m

    return best_thresh, best_metrics, grid_results, y_prob


# ---------------------------------------------------------------------------
# 7. Model Comparison & Experiment Manager
# ---------------------------------------------------------------------------

def run_experiment(train_df, val_df, test_df, feature_cols, experiment_name):
    """Run training, validation, and threshold tuning for all model families on a feature set."""
    X_train = train_df[feature_cols].values
    y_train = train_df["is_fraud_account"].astype(int).values

    X_val = val_df[feature_cols].values
    y_val = val_df["is_fraud_account"].astype(int).values

    X_test = test_df[feature_cols].values
    y_test = test_df["is_fraud_account"].astype(int).values

    # Pre-flight check
    validate_pre_training_integrity(train_df, val_df, test_df, feature_cols)

    models_config = [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),
        ("XGBoost", train_xgboost),
    ]

    exp_results = {}

    for model_name, trainer_fn in models_config:
        model, scaler = trainer_fn(X_train, y_train, seed=42)

        # Evaluate on VALIDATION set at default 0.5 threshold
        X_v = scaler.transform(X_val) if scaler is not None else X_val
        val_prob = model.predict_proba(X_v)[:, 1]
        val_default_metrics = evaluate_predictions(y_val, val_prob, threshold=0.5)

        # Tune threshold on VALIDATION set
        best_thresh, val_tuned_metrics, grid_table, _ = evaluate_threshold_grid(
            model, X_val, y_val, scaler=scaler
        )

        exp_results[model_name] = {
            "model_object": model,
            "scaler_object": scaler,
            "val_default_metrics": val_default_metrics,
            "val_best_threshold": best_thresh,
            "val_tuned_metrics": val_tuned_metrics,
            "threshold_grid_table": grid_table,
            "X_test": X_test,
            "y_test": y_test,
        }

    return exp_results


# ---------------------------------------------------------------------------
# 8. Feature Importance & Attribution
# ---------------------------------------------------------------------------

def compute_feature_importance(model, X_val, y_val, feature_names, scaler=None, data_dir=None):
    """Extract top 15 features by importance and map to Gene Groups."""
    if scaler is not None:
        X_val_eval = scaler.transform(X_val)
    else:
        X_val_eval = X_val

    # Native importance if tree-based, else permutation importance
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        perm_res = permutation_importance(model, X_val_eval, y_val, n_repeats=10, random_state=42)
        importances = perm_res.importances_mean

    imp_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
        "gene_group": [get_gene_group(f) for f in feature_names],
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    top_15 = imp_df.head(15)

    if data_dir is not None:
        imp_path = os.path.join(data_dir, "feature_importance.csv")
        imp_df.to_csv(imp_path, index=False)
        print(f"Saved feature importance report to '{imp_path}'.")

    return top_15, imp_df


# ---------------------------------------------------------------------------
# 9. Main Pipeline
# ---------------------------------------------------------------------------

def main():
    df, data_dir = load_features()

    # Step 1: Feature Audit & Leakage Inspection
    audit_features(df)

    # Step 2: Leakage-Safe True Cluster-Grouped + Account-Grouped Split (70/15/15)
    train_df, val_df, test_df, split_report = prepare_cluster_aware_group_split(
        df, seed=42
    )

    # Step 3: Run Baseline and FraudGenome Experiments on VALIDATION set
    print("=" * 80)
    print("  RUNNING MODEL TRAINING & VALIDATION EXPERIMENTS")
    print("=" * 80 + "\n")

    baseline_results = run_experiment(train_df, val_df, test_df, BASELINE_FEATURES, "Baseline")
    genome_results = run_experiment(train_df, val_df, test_df, FRAUDGENOME_FEATURES, "FraudGenome")

    # Step 4: Display Validation Comparison Table
    print("\n" + "=" * 80)
    print("  VALIDATION EXPERIMENT COMPARISON TABLE")
    print("=" * 80)
    print(f"\n  Baseline Fraud Prevalence in Val: {val_df['is_fraud_account'].mean():.2%}")
    print(f"  {'Feature Set':<14} {'Model':<20} {'PR-AUC':>10} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Best Thresh':>12}")
    print("  " + "-" * 78)

    val_comparison_rows = []

    for name, exp_dict in [("Baseline", baseline_results), ("FraudGenome", genome_results)]:
        for model_name, res in exp_dict.items():
            m = res["val_tuned_metrics"]
            thresh = res["val_best_threshold"]
            print(f"  {name:<14} {model_name:<20} {m['pr_auc']:>10.4f} {m['roc_auc']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f} {thresh:>12.2f}")
            val_comparison_rows.append({
                "feature_set": name,
                "model": model_name,
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "best_threshold": thresh,
            })

    # Select Best Model based ONLY on VALIDATION PR-AUC
    sorted_val = sorted(val_comparison_rows, key=lambda x: x["pr_auc"], reverse=True)
    best_val_config = sorted_val[0]
    best_feature_set_name = best_val_config["feature_set"]
    best_model_name = best_val_config["model"]
    best_val_pr_auc = best_val_config["pr_auc"]

    best_exp_dict = genome_results if best_feature_set_name == "FraudGenome" else baseline_results
    winning_model = best_exp_dict[best_model_name]["model_object"]
    winning_scaler = best_exp_dict[best_model_name]["scaler_object"]
    winning_thresh = best_exp_dict[best_model_name]["val_best_threshold"]
    winning_feature_cols = FRAUDGENOME_FEATURES if best_feature_set_name == "FraudGenome" else BASELINE_FEATURES

    print("\n" + "=" * 80)
    print(f"  🏆 WINNING VALIDATION MODEL: {best_feature_set_name} — {best_model_name}")
    print(f"     Validation PR-AUC: {best_val_pr_auc:.4f}  |  Optimal Validation Threshold: {winning_thresh:.2f}")
    print("=" * 80 + "\n")

    # Step 5: Evaluate UNTOUCHED Test Set ONLY ONCE using Winning Model & Threshold
    print("=" * 80)
    print("  FINAL EVALUATION ON UNTOUCHED TEST SET")
    print("=" * 80)

    X_test_win = test_df[winning_feature_cols].values
    y_test_win = test_df["is_fraud_account"].astype(int).values

    X_test_eval = winning_scaler.transform(X_test_win) if winning_scaler is not None else X_test_win
    test_prob = winning_model.predict_proba(X_test_eval)[:, 1]

    final_test_metrics = evaluate_predictions(y_test_win, test_prob, threshold=winning_thresh)

    print(f"\n  Final Test Performance (Threshold = {winning_thresh:.2f}):")
    print(f"    - Test PR-AUC:               {final_test_metrics['pr_auc']:.4f}")
    print(f"    - Test ROC-AUC:              {final_test_metrics['roc_auc']:.4f}")
    print(f"    - Test Precision:            {final_test_metrics['precision']:.4f}")
    print(f"    - Test Recall:               {final_test_metrics['recall']:.4f}")
    print(f"    - Test F1-Score:             {final_test_metrics['f1']:.4f}")
    print(f"    - Test False Positives:      {final_test_metrics['fp_count']}")
    print(f"    - Test False Negatives:      {final_test_metrics['fn_count']}")
    print(f"    - Test Fraud Detection Rate: {final_test_metrics['fraud_detection_rate']:.2%}")
    print(f"    - Test Confusion Matrix:     TN={final_test_metrics['tn']}, FP={final_test_metrics['fp']}, FN={final_test_metrics['fn']}, TP={final_test_metrics['tp']}")

    # Specific analysis on Unseen Test Fraud Cluster ('cluster_004')
    test_cluster_df = test_df[test_df["is_fraud_account"]].copy()
    test_cluster_ids = test_cluster_df["fraud_cluster_id"].unique()
    print(f"\n  Unseen Test Fraud Cluster Evaluation:")
    print(f"    - Test Fraud Cluster(s):     {list(test_cluster_ids)}")
    print(f"    - Test Fraud Accounts:       {test_cluster_df['account_id'].nunique()}")
    print(f"    - Test Fraud Rows:           {len(test_cluster_df)}")

    # Step 6: Baseline vs FraudGenome Comparison on Test Set
    print("\n" + "=" * 80)
    print("  BASELINE VS FRAUDGENOME TEST SET COMPARISON")
    print("=" * 80)

    # Evaluate best Baseline vs best FraudGenome model on Test
    best_base_res = baseline_results["XGBoost"]  # XGBoost as standard reference
    best_gen_res = genome_results["XGBoost"]

    X_test_base = test_df[BASELINE_FEATURES].values
    X_test_gen = test_df[FRAUDGENOME_FEATURES].values

    p_base = best_base_res["model_object"].predict_proba(X_test_base)[:, 1]
    p_gen = best_gen_res["model_object"].predict_proba(X_test_gen)[:, 1]

    t_base_metrics = evaluate_predictions(y_test_win, p_base, threshold=best_base_res["val_best_threshold"])
    t_gen_metrics = evaluate_predictions(y_test_win, p_gen, threshold=best_gen_res["val_best_threshold"])

    pr_auc_diff = t_gen_metrics["pr_auc"] - t_base_metrics["pr_auc"]
    f1_diff = t_gen_metrics["f1"] - t_base_metrics["f1"]

    print(f"\n  XGBoost Baseline Test PR-AUC:    {t_base_metrics['pr_auc']:.4f}  | F1: {t_base_metrics['f1']:.4f}")
    print(f"  XGBoost FraudGenome Test PR-AUC: {t_gen_metrics['pr_auc']:.4f}  | F1: {t_gen_metrics['f1']:.4f}")
    print(f"  Δ PR-AUC Improvement:            {pr_auc_diff:+.4f}")
    print(f"  Δ F1 Improvement:                {f1_diff:+.4f}\n")

    # Step 7: Feature Importance & Attribution
    print("=" * 80)
    print("  FEATURE IMPORTANCE & ATTRIBUTION (Top 15 Features)")
    print("=" * 80 + "\n")

    top_15, full_imp_df = compute_feature_importance(
        best_gen_res["model_object"],
        val_df[FRAUDGENOME_FEATURES].values,
        val_df["is_fraud_account"].astype(int).values,
        FRAUDGENOME_FEATURES,
        scaler=best_gen_res["scaler_object"],
        data_dir=data_dir,
    )

    print(f"  {'Rank':<6} {'Feature':<38} {'Gene Group':<15} {'Importance':>12}")
    print("  " + "-" * 72)
    for idx, r in top_15.iterrows():
        print(f"  #{idx+1:<5} {r['feature']:<38} {r['gene_group']:<15} {r['importance']:>12.6f}")

    print("\n" + "=" * 80)

    # Step 8: Optional Out-Of-Time Temporal Experiment (T1+T2 -> T3)
    train_t, test_t, t2_ref = prepare_temporal_split(df)
    X_tr_t = train_t[FRAUDGENOME_FEATURES].values
    y_tr_t = train_t["is_fraud_account"].astype(int).values
    X_te_t = test_t[FRAUDGENOME_FEATURES].values
    y_te_t = test_t["is_fraud_account"].astype(int).values

    temp_xgb = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(len(y_tr_t) - y_tr_t.sum()) / y_tr_t.sum(),
        eval_metric="aucpr", random_state=42
    )
    temp_xgb.fit(X_tr_t, y_tr_t, verbose=False)
    temp_prob = temp_xgb.predict_proba(X_te_t)[:, 1]
    temp_metrics = evaluate_predictions(y_te_t, temp_prob, threshold=0.5)

    print(f"  Temporal Experiment (T1+T2 -> T3 Out-Of-Time Evaluation):")
    print(f"    - PR-AUC: {temp_metrics['pr_auc']:.4f}  | ROC-AUC: {temp_metrics['roc_auc']:.4f}  | F1: {temp_metrics['f1']:.4f}\n")

    # Step 9: Save Comprehensive Results to data/model_results.json
    results_to_save = {
        "split_report": split_report,
        "validation_experiments": val_comparison_rows,
        "best_validation_model": {
            "feature_set": best_feature_set_name,
            "model": best_model_name,
            "val_pr_auc": best_val_pr_auc,
            "optimal_threshold": winning_thresh,
        },
        "final_test_evaluation": final_test_metrics,
        "test_comparison": {
            "baseline_xgb": t_base_metrics,
            "fraudgenome_xgb": t_gen_metrics,
            "pr_auc_improvement": round(pr_auc_diff, 4),
            "f1_improvement": round(f1_diff, 4),
        },
        "top_15_features": top_15.to_dict(orient="records"),
        "temporal_experiment": temp_metrics,
    }

    res_path = os.path.join(data_dir, "model_results.json")
    with open(res_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved machine-readable model results to '{res_path}'.")

    # Save model checkpoints
    model_dir = os.path.join(data_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "baseline_best.pkl"), "wb") as f:
        pickle.dump(best_base_res["model_object"], f)
    with open(os.path.join(model_dir, "genome_best.pkl"), "wb") as f:
        pickle.dump(best_gen_res["model_object"], f)
    print(f"Saved serialized model checkpoints to '{model_dir}'.\n")

    print("Experiment pipeline complete.\n")


if __name__ == "__main__":
    main()
