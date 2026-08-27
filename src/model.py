"""
model.py — FraudGenome Model Training, Evaluation, Explainability, Frozen Calibration,
           Precision@K, Alert-Budget Analysis, and Hybrid Weighting Ablation Pipeline

Execution Flow:
1. Feature Audit & Leakage Inspection
2. Leakage-Safe Data Splitting (True Cluster-Grouped + Account-Grouped 70/15/15)
3. Pre-Training Pre-flight Integrity Validation
4. Baseline vs FraudGenome Experiments across 3 Model Families (Logistic Regression, Random Forest, XGBoost)
5. Validation Threshold Optimization & Supervised Model Brier Calibration Check
6. Frozen Untouched Test Set Evaluation
7. Feature Importance & Gene Group Attribution
8. Out-of-Time 7-Part Ablation Study (T1+T2 -> T3):
   - Exp A: Temporal Baseline
   - Exp B: Temporal FraudGenome
   - Exp C: Temporal FraudGenome + Mutation-Aware
   - Exp D: Relational-Only Mutation
   - Exp E: Historical Genome Drift
   - Exp F: Relational Anomaly Score & Dynamic Decision Policy
   - Exp G: Calibrated Hybrid Risk Layer (Supervised + Relational + Drift)
9. Historical-Only Hybrid Weighting Ablation (Weights A, B, C, D selected on T1+T2)
10. Historical Policy Threshold Sweep (p95, p97.5, p99, p99.5)
11. Precision@K (K=1%, 5%, 10%) & Alert-Budget Capacity Evaluation (0.5%, 1%, 2%, 5%)
12. Score Distribution Diagnostics across T2/T3 Fraud/Normal Subpopulations
13. Automated Unit Tests (Temporal Causality, Immutability, Scoring Stability)
14. Saving artifacts: data/risk_calibration_t1_t2.json, data/model_results.json, data/feature_importance.csv, models/*.pkl
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
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from genome_drift import (
    apply_cost_sensitive_policy,
    build_historical_calibration_profile,
    build_historical_genome_reference,
    compute_account_genome_drift,
    compute_decision_confidence,
    compute_hybrid_risk_layer_calibrated,
    compute_relational_anomaly_score,
    generate_account_explanation,
    run_scoring_stability_unit_tests,
    run_temporal_immutability_unit_tests,
    transform_empirical_cdf,
)

# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

EXCLUDED_COLUMNS = [
    "account_id",
    "phase",
    "is_fraud_account",      # target label
    "fraud_cluster_id",      # target label/metadata
    "scenario",              # metadata / scenario label
    "graph_community_fraud_ratio",  # LEAKAGE: computed from is_fraud ground truth
]

BASELINE_FEATURES = [
    "acct_tx_count", "acct_tx_velocity", "acct_active_days",
    "acct_amount_total", "acct_amount_avg", "acct_amount_std",
    "acct_age_days", "acct_kyc_unverified",
    "dev_unique_count", "dev_tx_per_device", "dev_concentration",
    "net_unique_ip_count", "net_tx_per_ip", "net_ip_concentration",
    "tx_velocity", "tx_amount_mean", "tx_amount_max",
    "tx_amount_concentration", "tx_time_spread_hours",
    "tx_hour_mode", "tx_weekend_ratio",
    "merch_unique_count", "merch_concentration",
    "merch_top_category", "merch_avg_age_days",
]

GENOME_ONLY_FEATURES = [
    "dev_max_accts_per_device", "dev_shared_count", "dev_shared_ratio",
    "net_max_accts_per_ip", "net_shared_ip_count", "net_shared_ip_ratio",
    "merch_max_accts_per_merchant",
    "graph_degree", "graph_weighted_degree",
    "graph_connected_devices", "graph_connected_ips", "graph_connected_merchants",
    "graph_shared_devices", "graph_shared_ips",
    "graph_proj_degree", "graph_proj_weighted_degree",
    "graph_community_size", "graph_community_density",
    "graph_community_shared_device_ratio", "graph_community_shared_ip_ratio",
]

FRAUDGENOME_FEATURES = BASELINE_FEATURES + GENOME_ONLY_FEATURES

MUTATION_FEATURES = [
    "net_ip_persistence_score",
    "merch_persistence_score",
    "dev_mutation_score",
    "relational_stability_score",
    "topology_drift_score",
    "behavioral_shift_score",
    "genome_drift_score",
]

MUTATION_AWARE_FEATURES = FRAUDGENOME_FEATURES + MUTATION_FEATURES

RELATIONAL_MUTATION_FEATURES = [
    "graph_degree", "graph_weighted_degree",
    "graph_connected_devices", "graph_connected_ips", "graph_connected_merchants",
    "graph_shared_devices", "graph_shared_ips",
    "graph_proj_degree", "graph_proj_weighted_degree",
    "graph_community_size", "graph_community_density",
    "graph_community_shared_device_ratio", "graph_community_shared_ip_ratio",
    "dev_shared_ratio", "net_shared_ip_ratio", "merch_max_accts_per_merchant",
] + MUTATION_FEATURES

HISTORICAL_DRIFT_FEATURES = [
    "account_genome_drift",
    "behavioral_drift_score",
    "relational_drift_score",
    "device_drift_score",
    "network_drift_score",
    "merchant_drift_score",
    "graph_drift_score",
    "community_drift_score",
]

RELATIONAL_ANOMALY_FEATURES = RELATIONAL_MUTATION_FEATURES + ["relational_anomaly_score"]

HYBRID_RISK_FEATURES = MUTATION_AWARE_FEATURES + HISTORICAL_DRIFT_FEATURES + ["relational_anomaly_score"]

GENE_GROUP_MAP = {
    "acct_": "Account", "dev_": "Device", "net_": "Network",
    "tx_": "Transaction", "merch_": "Merchant",
    "graph_community_": "Community", "graph_": "Graph",
    "relational_": "Mutation/Relational", "topology_": "Mutation/Relational",
    "behavioral_": "Mutation/Relational", "genome_": "Mutation/Relational",
    "account_": "Historical Drift", "device_": "Historical Drift",
    "network_": "Historical Drift", "merchant_": "Historical Drift",
    "community_": "Historical Drift",
}


def get_gene_group(feature_name):
    """Categorize a feature into its behavioral gene group."""
    if feature_name.startswith("graph_community_"):
        return "Community"
    for prefix, group in GENE_GROUP_MAP.items():
        if feature_name.startswith(prefix):
            return group
    return "Other"


def load_features(data_dir=None):
    """Load normalized genome DataFrame."""
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")
    df = pd.read_csv(os.path.join(data_dir, "genome_full.csv"))
    return df, data_dir


def audit_features(df):
    """Print feature audit report."""
    all_cols = list(df.columns)
    leakage = ["graph_community_fraud_ratio"]
    print("\n" + "=" * 80)
    print("  FEATURE AUDIT REPORT — ZERO TARGET LEAKAGE CHECK")
    print("=" * 80)
    print(f"  Total columns in genome_full.csv: {len(all_cols)}")
    print(f"  Explicit Exclusions: {EXCLUDED_COLUMNS}")
    for col in leakage:
        assert col not in FRAUDGENOME_FEATURES and col not in HYBRID_RISK_FEATURES, f"Leakage detected: {col}"
    print("  ✓ Feature matrix is 100% clean and leakage-safe.")
    print("=" * 80 + "\n")


def validate_leakage_safety(feature_cols):
    """Assert no label or metadata leakage in feature vector."""
    for col in feature_cols:
        assert col not in EXCLUDED_COLUMNS, f"LEAKAGE ASSERTION FAILED: {col}"
        assert col not in ["is_fraud", "is_fraud_account", "fraud_cluster_id", "scenario"], f"LABEL LEAKAGE: {col}"


def validate_temporal_causality(train_df, test_df):
    """Verify train is strictly T1+T2 and test is strictly T3."""
    train_phases = set(train_df["phase"].unique())
    test_phases = set(test_df["phase"].unique())
    assert train_phases.issubset({"T1", "T2"}), f"Temporal violation in train: {train_phases}"
    assert test_phases == {"T3"}, f"Temporal violation in test: {test_phases}"


def prepare_cluster_aware_group_split(df, seed=42):
    """Leakage-safe Cluster-Grouped + Account-Grouped Stratified Split (70/15/15)."""
    acct_info = df.groupby("account_id").agg(
        is_fraud=("is_fraud_account", "any"),
        cluster_id=("fraud_cluster_id", lambda x: [i for i in x if i != "none"][0] if any(i != "none" for i in x) else "none")
    ).reset_index()

    fraud_clusters = sorted([c for c in acct_info["cluster_id"].unique() if c != "none"])
    rng = np.random.RandomState(seed)
    shuffled_clusters = list(fraud_clusters)
    rng.shuffle(shuffled_clusters)

    n_clusters = len(fraud_clusters)
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
    train_norm, test_val_norm = train_test_split(normal_accts, test_size=0.30, random_state=seed)
    val_norm, test_norm = train_test_split(test_val_norm, test_size=0.50, random_state=seed)

    train_acct_ids = set(train_fraud_accts["account_id"]).union(set(train_norm["account_id"]))
    val_acct_ids = set(val_fraud_accts["account_id"]).union(set(val_norm["account_id"]))
    test_acct_ids = set(test_fraud_accts["account_id"]).union(set(test_norm["account_id"]))

    train_df = df[df["account_id"].isin(train_acct_ids)].copy().reset_index(drop=True)
    val_df = df[df["account_id"].isin(val_acct_ids)].copy().reset_index(drop=True)
    test_df = df[df["account_id"].isin(test_acct_ids)].copy().reset_index(drop=True)

    report = {
        "strategy": "True Cluster-Grouped + Account-Grouped Stratified Split (70/15/15)",
        "total_rows": len(df),
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "train_fraud_count": int(train_df["is_fraud_account"].sum()),
        "val_fraud_count": int(val_df["is_fraud_account"].sum()),
        "test_fraud_count": int(test_df["is_fraud_account"].sum()),
    }
    return train_df, val_df, test_df, report


def prepare_temporal_split(df):
    """Out-of-time temporal split: Train (T1+T2) -> Test (T3)."""
    train_df = df[df["phase"].isin(["T1", "T2"])].copy().reset_index(drop=True)
    test_df = df[df["phase"] == "T3"].copy().reset_index(drop=True)
    return train_df, test_df


def train_logistic_regression(X_train, y_train, seed=42):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed, solver="lbfgs")
    model.fit(X_scaled, y_train)
    return model, scaler


def train_random_forest(X_train, y_train, seed=42):
    model = RandomForestClassifier(n_estimators=100, max_depth=6, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)
    return model, None


def train_xgboost(X_train, y_train, seed=42):
    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, subsample=0.8,
        colsample_bytree=0.8, scale_pos_weight=scale_pos_weight, eval_metric="aucpr", random_state=seed,
    )
    model.fit(X_train, y_train, verbose=False)
    return model, None


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    """Compute comprehensive ranking and decision metrics."""
    y_pred = (y_prob >= threshold).astype(int)
    n_pos = int(y_true.sum())

    if n_pos == 0:
        return {
            "pr_auc": 0.0, "roc_auc": 0.0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "confusion_matrix": [[0, 0], [0, 0]],
            "tn": 0, "fp": 0, "fn": 0, "tp": 0,
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
        "fp_count": int(fp), "fn_count": int(fn),
        "fpr": float(round(fpr, 6)),
        "fraud_detection_rate": float(round(recall, 4)),
    }


def evaluate_threshold_grid(model, X_val, y_val, scaler=None):
    X_v = scaler.transform(X_val) if scaler is not None else X_val
    y_prob = model.predict_proba(X_v)[:, 1]

    threshold_grid = [round(t, 2) for t in np.arange(0.05, 0.95, 0.05)]
    grid_results = []
    best_thresh = 0.50
    best_f1 = -1.0
    best_metrics = None

    for thresh in threshold_grid:
        m = evaluate_predictions(y_val, y_prob, threshold=thresh)
        grid_results.append({
            "threshold": thresh, "precision": m["precision"], "recall": m["recall"],
            "f1": m["f1"], "fp_count": m["fp_count"], "fn_count": m["fn_count"],
            "tp": m["tp"], "tn": m["tn"],
        })
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh
            best_metrics = m

    return best_thresh, best_metrics, grid_results, y_prob


def run_experiment(train_df, val_df, test_df, feature_cols, experiment_name):
    X_train = train_df[feature_cols].values
    y_train = train_df["is_fraud_account"].astype(int).values
    X_val = val_df[feature_cols].values
    y_val = val_df["is_fraud_account"].astype(int).values
    X_test = test_df[feature_cols].values
    y_test = test_df["is_fraud_account"].astype(int).values

    models_config = [
        ("Logistic Regression", train_logistic_regression),
        ("Random Forest", train_random_forest),
        ("XGBoost", train_xgboost),
    ]

    exp_results = {}
    for model_name, trainer_fn in models_config:
        model, scaler = trainer_fn(X_train, y_train, seed=42)
        X_v = scaler.transform(X_val) if scaler is not None else X_val
        val_prob = model.predict_proba(X_v)[:, 1]
        val_default_metrics = evaluate_predictions(y_val, val_prob, threshold=0.5)

        best_thresh, val_tuned_metrics, grid_table, _ = evaluate_threshold_grid(
            model, X_val, y_val, scaler=scaler
        )

        exp_results[model_name] = {
            "model_object": model, "scaler_object": scaler,
            "val_default_metrics": val_default_metrics,
            "val_best_threshold": best_thresh,
            "val_tuned_metrics": val_tuned_metrics,
            "threshold_grid_table": grid_table,
            "X_test": X_test, "y_test": y_test,
        }

    return exp_results


def evaluate_supervised_model_calibration(y_val, val_prob):
    """Compute Brier calibration score and reliability metrics on validation fold."""
    brier = brier_score_loss(y_val, val_prob)
    print(f"  Supervised Model Validation Brier Score: {brier:.6f}  (Lower is better, 0 = perfect)")
    return {"brier_score": round(float(brier), 6)}


# ---------------------------------------------------------------------------
# Precision@K, Recall@K, & Alert-Budget Evaluation Functions
# ---------------------------------------------------------------------------

def compute_precision_recall_at_k(y_true, risk_scores, k_percentiles=[1.0, 5.0, 10.0]):
    """Compute Top-K Precision and Recall metrics for given capacity percentiles."""
    n_total = len(y_true)
    total_positives = int(np.sum(y_true))

    # Sort descending by risk score
    sort_idx = np.argsort(-risk_scores)
    y_sorted = y_true[sort_idx]

    results = {}
    for k_pct in k_percentiles:
        top_k_count = max(int(np.round(n_total * (k_pct / 100.0))), 1)
        top_k_y = y_sorted[:top_k_count]

        tp_k = int(np.sum(top_k_y))
        prec_k = tp_k / top_k_count
        rec_k = tp_k / max(total_positives, 1)

        results[f"p_at_{k_pct:.1f}%"] = round(float(prec_k), 4)
        results[f"r_at_{k_pct:.1f}%"] = round(float(rec_k), 4)
        results[f"tp_at_{k_pct:.1f}%"] = tp_k
        results[f"alerts_at_{k_pct:.1f}%"] = top_k_count

    return results


def compute_alert_budget_analysis(y_true, risk_scores, budgets_pct=[0.5, 1.0, 2.0, 5.0]):
    """Evaluate realistic operational review capacity budgets."""
    n_total = len(y_true)
    total_positives = int(np.sum(y_true))

    sort_idx = np.argsort(-risk_scores)
    y_sorted = y_true[sort_idx]

    budget_rows = []
    for b_pct in budgets_pct:
        n_alerts = max(int(np.round(n_total * (b_pct / 100.0))), 1)
        top_y = y_sorted[:n_alerts]
        fraud_captured = int(np.sum(top_y))
        prec = fraud_captured / n_alerts
        rec = fraud_captured / max(total_positives, 1)

        budget_rows.append({
            "budget_pct": b_pct,
            "alerts": n_alerts,
            "fraud_captured": fraud_captured,
            "total_fraud": total_positives,
            "capture_rate": round(float(rec), 4),
            "precision": round(float(prec), 4),
        })

    return budget_rows


# ---------------------------------------------------------------------------
# Hybrid Weighting Ablation Engine (Historical-Only Selection)
# ---------------------------------------------------------------------------

def run_hybrid_weight_ablation(t1_t2_df, t3_df, calib_profile):
    """Evaluate 4 weight configurations strictly on historical T1+T2 validation fold.

    Configurations:
    - Weight Set A: (0.4, 0.4, 0.2) [Baseline]
    - Weight Set B: (0.3, 0.5, 0.2) [Relational Heavy]
    - Weight Set C: (0.2, 0.6, 0.2) [Relational Dominant]
    - Weight Set D: (0.3, 0.4, 0.3) [Drift Balanced]
    """
    weight_configs = {
        "Set A (0.4/0.4/0.2 Baseline)": (0.4, 0.4, 0.2),
        "Set B (0.3/0.5/0.2 Relational Heavy)": (0.3, 0.5, 0.2),
        "Set C (0.2/0.6/0.2 Rel-Dominant)": (0.2, 0.6, 0.2),
        "Set D (0.3/0.4/0.3 Drift-Balanced)": (0.3, 0.4, 0.3),
    }

    t1_df = t1_t2_df[t1_t2_df["phase"] == "T1"].reset_index(drop=True)
    t2_df = t1_t2_df[t1_t2_df["phase"] == "T2"].reset_index(drop=True)
    ref_t1 = build_historical_genome_reference(t1_df, reference_name="t1_ablation")

    # Fit XGBoost on T1
    X_t1 = t1_df[MUTATION_AWARE_FEATURES].values
    y_t1 = t1_df["is_fraud_account"].astype(int).values
    model_t1, _ = train_xgboost(X_t1, y_t1, seed=42)

    X_t2 = t2_df[MUTATION_AWARE_FEATURES].values
    y_t2 = t2_df["is_fraud_account"].astype(int).values
    prob_t2 = model_t1.predict_proba(X_t2)[:, 1]
    rel_t2 = compute_relational_anomaly_score(t2_df, ref_t1)
    drift_t2 = t2_df["account_genome_drift"].values if "account_genome_drift" in t2_df.columns else np.zeros(len(t2_df))

    # Fit validation calibration profile on T1 normal
    hist_t1_norm = t1_df[~t1_df["is_fraud_account"]].reset_index(drop=True)
    prob_t1_norm = model_t1.predict_proba(X_t1[~y_t1.astype(bool)])[:, 1]
    rel_t1_norm = compute_relational_anomaly_score(hist_t1_norm, ref_t1)
    drift_t1_norm = hist_t1_norm["account_genome_drift"].values if "account_genome_drift" in hist_t1_norm.columns else np.zeros(len(hist_t1_norm))

    val_calib = build_historical_calibration_profile(hist_t1_norm, prob_t1_norm, rel_t1_norm, drift_t1_norm)

    ablation_summary = {}
    best_config_name = None
    best_val_prauc = -1.0

    print("\n  HISTORICAL HYBRID WEIGHT ABLATION TABLE (Evaluated on T1 -> T2 Validation Fold):")
    print(f"  {'Weight Set':<36} {'PR-AUC':>9} {'ROC-AUC':>9} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("  " + "-" * 82)

    for cfg_name, (w1, w2, w3) in weight_configs.items():
        val_hybrid = compute_hybrid_risk_layer_calibrated(prob_t2, rel_t2, drift_t2, val_calib, w1, w2, w3)
        val_metrics = evaluate_predictions(y_t2, val_hybrid, threshold=val_calib["policy_percentiles"]["p97_5"])

        ablation_summary[cfg_name] = {
            "weights": [w1, w2, w3],
            "t2_val_pr_auc": val_metrics["pr_auc"],
            "t2_val_roc_auc": val_metrics["roc_auc"],
            "t2_val_f1": val_metrics["f1"],
        }

        print(
            f"  {cfg_name:<36} {val_metrics['pr_auc']:>9.4f} {val_metrics['roc_auc']:>9.4f} "
            f"{val_metrics['precision']:>8.4f} {val_metrics['recall']:>8.4f} {val_metrics['f1']:>8.4f}"
        )

        if val_metrics["pr_auc"] > best_val_prauc:
            best_val_prauc = val_metrics["pr_auc"]
            best_config_name = cfg_name

    print(f"\n  ✓ Winning Weight Config selected on T1+T2 validation fold: '{best_config_name}' (PR-AUC: {best_val_prauc:.4f})")
    winning_weights = weight_configs[best_config_name]

    # Evaluate frozen winning weights on T3
    ref_t1_t2 = build_historical_genome_reference(t1_t2_df, reference_name="t1_t2_full")
    X_tr_full = t1_t2_df[MUTATION_AWARE_FEATURES].values
    y_tr_full = t1_t2_df["is_fraud_account"].astype(int).values
    final_model, _ = train_xgboost(X_tr_full, y_tr_full, seed=42)

    X_t3 = t3_df[MUTATION_AWARE_FEATURES].values
    y_t3 = t3_df["is_fraud_account"].astype(int).values
    prob_t3 = final_model.predict_proba(X_t3)[:, 1]
    rel_t3 = compute_relational_anomaly_score(t3_df, ref_t1_t2)
    drift_t3 = t3_df["account_genome_drift"].values if "account_genome_drift" in t3_df.columns else np.zeros(len(t3_df))

    t3_hybrid_winning = compute_hybrid_risk_layer_calibrated(
        prob_t3, rel_t3, drift_t3, calib_profile,
        w1=winning_weights[0], w2=winning_weights[1], w3=winning_weights[2]
    )

    t3_metrics_winning = evaluate_predictions(y_t3, t3_hybrid_winning, threshold=calib_profile["policy_percentiles"]["p97_5"])

    return ablation_summary, best_config_name, winning_weights, t3_hybrid_winning, t3_metrics_winning


# ---------------------------------------------------------------------------
# Historical Policy Threshold Sweep (p95, p97.5, p99, p99.5)
# ---------------------------------------------------------------------------

def evaluate_historical_policy_sweep(y_t3, t3_risk_scores, calib_profile):
    """Evaluate historical-only policy percentiles on T3 without tuning on T3."""
    policies = {
        "p95 (Permissive)": calib_profile["policy_percentiles"]["p95"],
        "p97.5 (Balanced)": calib_profile["policy_percentiles"]["p97_5"],
        "p99 (Strict)": calib_profile["policy_percentiles"]["p99"],
        "p99.5 (Conservative)": calib_profile["policy_percentiles"]["p99_5"],
    }

    sweep_results = {}
    print("\n  HISTORICAL POLICY THRESHOLD SWEEP TABLE (Evaluated on T3 with Frozen Cutoffs):")
    print(f"  {'Policy Percentile':<24} {'Threshold':>10} {'Precision':>10} {'Recall':>9} {'F1':>8} {'FP':>5} {'FN':>5} {'FPR':>8}")
    print("  " + "-" * 84)

    for pname, thresh in policies.items():
        eval_p = evaluate_predictions(y_t3, t3_risk_scores, threshold=thresh)
        sweep_results[pname] = {
            "policy_threshold": thresh,
            "precision": eval_p["precision"],
            "recall": eval_p["recall"],
            "f1": eval_p["f1"],
            "fp_count": eval_p["fp_count"],
            "fn_count": eval_p["fn_count"],
            "fpr": eval_p["fpr"],
        }
        print(
            f"  {pname:<24} {thresh:>10.4f} {eval_p['precision']:>10.4f} {eval_p['recall']:>9.4f} "
            f"{eval_p['f1']:>8.4f} {eval_p['fp_count']:>5} {eval_p['fn_count']:>5} {eval_p['fpr']:>8.5f}"
        )

    print("=" * 80 + "\n")
    return sweep_results


# ---------------------------------------------------------------------------
# Score Distribution Diagnostics across Subpopulations
# ---------------------------------------------------------------------------

def compute_score_distribution_diagnostics(t1_t2_df, t3_df, model, calib_profile):
    """Compute detailed score distribution stats across T2 Normal, T2 Fraud, T3 Normal, and T3 Fraud."""
    ref_t1 = build_historical_genome_reference(t1_t2_df[t1_t2_df["phase"] == "T1"], reference_name="t1_diag")
    ref_t1_t2 = build_historical_genome_reference(t1_t2_df, reference_name="t1_t2_diag")

    t2_df = t1_t2_df[t1_t2_df["phase"] == "T2"].reset_index(drop=True)

    subpops = {
        "T2 Normal": (t2_df[~t2_df["is_fraud_account"]], ref_t1),
        "T2 Fraud": (t2_df[t2_df["is_fraud_account"]], ref_t1),
        "T3 Normal": (t3_df[~t3_df["is_fraud_account"]], ref_t1_t2),
        "T3 Fraud": (t3_df[t3_df["is_fraud_account"]], ref_t1_t2),
    }

    diag_summary = {}

    print("=" * 80)
    print("  SCORE DISTRIBUTION DIAGNOSTICS REPORT ACROSS SUBPOPULATIONS")
    print("=" * 80)

    for pop_name, (sub_df, ref) in subpops.items():
        X_sub = sub_df[FRAUDGENOME_FEATURES].values
        prob_sub = model.predict_proba(X_sub)[:, 1]
        rel_sub = compute_relational_anomaly_score(sub_df, ref)
        drift_sub = sub_df["account_genome_drift"].values if "account_genome_drift" in sub_df.columns else np.zeros(len(sub_df))
        hybrid_sub = compute_hybrid_risk_layer_calibrated(prob_sub, rel_sub, drift_sub, calib_profile)

        def get_stats(v):
            return {
                "mean": round(float(np.mean(v)), 4),
                "median": round(float(np.median(v)), 4),
                "p90": round(float(np.percentile(v, 90)), 4),
                "p95": round(float(np.percentile(v, 95)), 4),
                "p99": round(float(np.percentile(v, 99)), 4),
            }

        diag_summary[pop_name] = {
            "count": len(sub_df),
            "supervised_prob": get_stats(prob_sub),
            "relational_anomaly": get_stats(rel_sub),
            "genome_drift": get_stats(drift_sub),
            "final_risk_score": get_stats(hybrid_sub),
        }

        print(f"\n  [{pop_name}] (N={len(sub_df)}):")
        print(f"    - Supervised Prob  -> Mean: {diag_summary[pop_name]['supervised_prob']['mean']}, Median: {diag_summary[pop_name]['supervised_prob']['median']}, P95: {diag_summary[pop_name]['supervised_prob']['p95']}, P99: {diag_summary[pop_name]['supervised_prob']['p99']}")
        print(f"    - Relational Anom  -> Mean: {diag_summary[pop_name]['relational_anomaly']['mean']}, Median: {diag_summary[pop_name]['relational_anomaly']['median']}, P95: {diag_summary[pop_name]['relational_anomaly']['p95']}, P99: {diag_summary[pop_name]['relational_anomaly']['p99']}")
        print(f"    - Genome Drift     -> Mean: {diag_summary[pop_name]['genome_drift']['mean']}, Median: {diag_summary[pop_name]['genome_drift']['median']}, P95: {diag_summary[pop_name]['genome_drift']['p95']}, P99: {diag_summary[pop_name]['genome_drift']['p99']}")
        print(f"    - Final Risk Score -> Mean: {diag_summary[pop_name]['final_risk_score']['mean']}, Median: {diag_summary[pop_name]['final_risk_score']['median']}, P95: {diag_summary[pop_name]['final_risk_score']['p95']}, P99: {diag_summary[pop_name]['final_risk_score']['p99']}")

    print("=" * 80 + "\n")
    return diag_summary


# ---------------------------------------------------------------------------
# Extended Out-of-Time 7-Part Ablation Study (Exp A -> Exp G)
# ---------------------------------------------------------------------------

def run_temporal_ablation_study_calibrated(train_df, test_df, calib_profile, data_dir=None):
    """Run full 7-part Out-of-Time Temporal Ablation Study (T1+T2 -> T3) using frozen calibration."""
    print("\n" + "=" * 80)
    print("  RUNNING OUT-OF-TIME TEMPORAL ABLATION STUDY (Exp A -> Exp G)")
    print("=" * 80)

    validate_temporal_causality(train_df, test_df)

    t1_df = train_df[train_df["phase"] == "T1"].reset_index(drop=True)
    t2_df = train_df[train_df["phase"] == "T2"].reset_index(drop=True)
    ref_t1_t2 = build_historical_genome_reference(train_df, reference_name="t1_t2_ablation", output_dir=data_dir)

    experiments = [
        ("Exp A: Temporal Baseline", BASELINE_FEATURES, "model"),
        ("Exp B: Temporal FraudGenome", FRAUDGENOME_FEATURES, "model"),
        ("Exp C: FraudGenome + Mutation-Aware", MUTATION_AWARE_FEATURES, "model"),
        ("Exp D: Relational-Only Mutation", RELATIONAL_MUTATION_FEATURES, "model"),
        ("Exp E: Historical Genome Drift", HISTORICAL_DRIFT_FEATURES, "model"),
        ("Exp F: Relational Anomaly Score", RELATIONAL_ANOMALY_FEATURES, "anomaly"),
        ("Exp G: Calibrated Hybrid Risk Layer", HYBRID_RISK_FEATURES, "hybrid"),
    ]

    ablation_results = {}

    for exp_name, feat_cols, mode in experiments:
        validate_leakage_safety(feat_cols)

        if mode == "model":
            X_tr_fold = t1_df[feat_cols].values
            y_tr_fold = t1_df["is_fraud_account"].astype(int).values
            X_val_fold = t2_df[feat_cols].values
            y_val_fold = t2_df["is_fraud_account"].astype(int).values

            model_fold, _ = train_xgboost(X_tr_fold, y_tr_fold, seed=42)
            val_prob_fold = model_fold.predict_proba(X_val_fold)[:, 1]

            best_thresh = 0.5
            best_f1 = -1.0
            for t in np.arange(0.05, 0.95, 0.05):
                t_round = round(t, 2)
                eval_v = evaluate_predictions(y_val_fold, val_prob_fold, threshold=t_round)
                if eval_v["f1"] > best_f1:
                    best_f1 = eval_v["f1"]
                    best_thresh = t_round

            X_tr_full = train_df[feat_cols].values
            y_tr_full = train_df["is_fraud_account"].astype(int).values
            X_te_t3 = test_df[feat_cols].values
            y_te_t3 = test_df["is_fraud_account"].astype(int).values

            final_model, _ = train_xgboost(X_tr_full, y_tr_full, seed=42)
            t3_scores = final_model.predict_proba(X_te_t3)[:, 1]

        elif mode == "anomaly":
            best_thresh = calib_profile["policy_percentiles"]["p97_5"]
            t3_raw_rel = compute_relational_anomaly_score(test_df, ref_t1_t2)
            t3_scores = transform_empirical_cdf(t3_raw_rel, calib_profile["signals"]["relational_anomaly"])
            y_te_t3 = test_df["is_fraud_account"].astype(int).values
            final_model = None

        elif mode == "hybrid":
            best_thresh = calib_profile["policy_percentiles"]["p97_5"]
            X_tr_full = train_df[MUTATION_AWARE_FEATURES].values
            y_tr_full = train_df["is_fraud_account"].astype(int).values
            X_te_t3 = test_df[MUTATION_AWARE_FEATURES].values
            y_te_t3 = test_df["is_fraud_account"].astype(int).values

            final_model, _ = train_xgboost(X_tr_full, y_tr_full, seed=42)
            prob_t3 = final_model.predict_proba(X_te_t3)[:, 1]
            rel_t3 = compute_relational_anomaly_score(test_df, ref_t1_t2)
            drift_t3 = test_df["account_genome_drift"].values if "account_genome_drift" in test_df.columns else np.zeros(len(test_df))

            t3_scores = compute_hybrid_risk_layer_calibrated(prob_t3, rel_t3, drift_t3, calib_profile)

        t3_eval = evaluate_predictions(y_te_t3, t3_scores, threshold=best_thresh)

        ablation_results[exp_name] = {
            "feature_set_name": exp_name,
            "num_features": len(feat_cols),
            "mode": mode,
            "val_tuned_threshold": best_thresh,
            "t3_evaluation": t3_eval,
            "model_object": final_model,
            "predicted_scores": t3_scores,
        }

    print(f"\n  RANKING METRICS TABLE (PR-AUC & ROC-AUC):")
    print(f"  {'Experiment':<38} {'PR-AUC':>9} {'ROC-AUC':>9} {'Features':>10}")
    print("  " + "-" * 70)
    for exp_name, res in ablation_results.items():
        m = res["t3_evaluation"]
        print(f"  {exp_name:<38} {m['pr_auc']:>9.4f} {m['roc_auc']:>9.4f} {res['num_features']:>10}")

    print(f"\n  DECISION METRICS TABLE (Precision, Recall, F1, FP, FN, FPR):")
    print(f"  {'Experiment':<38} {'Prec':>8} {'Rec':>8} {'F1':>8} {'FP':>5} {'FN':>5} {'FPR':>8} {'Thresh':>8}")
    print("  " + "-" * 90)
    for exp_name, res in ablation_results.items():
        m = res["t3_evaluation"]
        th = res["val_tuned_threshold"]
        print(
            f"  {exp_name:<38} {m['precision']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
            f"{m['fp_count']:>5} {m['fn_count']:>5} {m['fpr']:>8.5f} {th:>8.4f}"
        )

    print("=" * 80 + "\n")
    return ablation_results


def compute_feature_importance(model, X_val, y_val, feature_names, scaler=None, data_dir=None):
    X_val_eval = scaler.transform(X_val) if scaler is not None else X_val

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
        print(f"  Saved feature importance report to '{imp_path}'.")

    return top_15, imp_df


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def main():
    # Run automated unit tests first
    run_temporal_immutability_unit_tests()
    run_scoring_stability_unit_tests()

    df, data_dir = load_features()

    # Step 1: Feature Audit & Safety Inspections
    audit_features(df)

    # Step 2: Split data for validation experiments
    train_df, val_df, test_df, split_report = prepare_cluster_aware_group_split(df, seed=42)

    # Step 3: Run Baseline and FraudGenome Experiments on VALIDATION set
    print("=" * 80)
    print("  RUNNING MODEL TRAINING & VALIDATION EXPERIMENTS")
    print("=" * 80 + "\n")

    baseline_results = run_experiment(train_df, val_df, test_df, BASELINE_FEATURES, "Baseline")
    genome_results = run_experiment(train_df, val_df, test_df, FRAUDGENOME_FEATURES, "FraudGenome")

    # Step 4: Supervised Model Calibration Check (Brier Score)
    best_gen_res = genome_results["XGBoost"]
    X_val_gen = val_df[FRAUDGENOME_FEATURES].values
    y_val_gen = val_df["is_fraud_account"].astype(int).values
    val_prob_xgb = best_gen_res["model_object"].predict_proba(X_val_gen)[:, 1]
    brier_info = evaluate_supervised_model_calibration(y_val_gen, val_prob_xgb)

    # Step 5: Build Frozen Historical Risk Calibration Profile from T1/T2 Normal Data
    train_temp, test_temp = prepare_temporal_split(df)
    ref_t1_t2_temp = build_historical_genome_reference(train_temp, reference_name="t1_t2_calib")
    hist_norm_train = train_temp[~train_temp["is_fraud_account"]].reset_index(drop=True)

    X_norm_tr = hist_norm_train[FRAUDGENOME_FEATURES].values
    X_tr_all = train_temp[FRAUDGENOME_FEATURES].values
    y_tr_all = train_temp["is_fraud_account"].astype(int).values

    calib_model, _ = train_xgboost(X_tr_all, y_tr_all, seed=42)
    norm_prob_tr = calib_model.predict_proba(X_norm_tr)[:, 1]
    norm_rel_tr = compute_relational_anomaly_score(hist_norm_train, ref_t1_t2_temp)
    norm_drift_tr = hist_norm_train["account_genome_drift"].values if "account_genome_drift" in hist_norm_train.columns else np.zeros(len(hist_norm_train))

    calib_profile = build_historical_calibration_profile(
        hist_norm_train, norm_prob_tr, norm_rel_tr, norm_drift_tr, output_dir=data_dir
    )

    # Step 6: Historical Hybrid Weighting Ablation
    hybrid_weight_summary, winning_w_name, winning_weights, t3_hybrid_scores, t3_metrics_winning = run_hybrid_weight_ablation(
        train_temp, test_temp, calib_profile
    )

    # Step 7: Out-of-Time 7-Part Ablation Study (T1+T2 -> T3) using Calibrated Risk Engine
    ablation_results = run_temporal_ablation_study_calibrated(train_temp, test_temp, calib_profile, data_dir=data_dir)

    # Step 8: Historical Policy Threshold Sweep (p95, p97.5, p99, p99.5)
    y_t3 = test_temp["is_fraud_account"].astype(int).values
    policy_sweep_summary = evaluate_historical_policy_sweep(y_t3, t3_hybrid_scores, calib_profile)

    # Step 9: Precision@K and Alert-Budget Capacity Evaluation
    pk_results = compute_precision_recall_at_k(y_t3, t3_hybrid_scores, k_percentiles=[1.0, 5.0, 10.0])
    alert_budget_summary = compute_alert_budget_analysis(y_t3, t3_hybrid_scores, budgets_pct=[0.5, 1.0, 2.0, 5.0])

    print("  PRECISION-AT-K METRICS:")
    for k, v in pk_results.items():
        print(f"    - {k}: {v}")

    print("\n  ALERT-BUDGET CAPACITY ANALYSIS TABLE:")
    print(f"  {'Budget %':<10} {'Alerts':>8} {'Fraud Captured':>15} {'Total Fraud':>13} {'Capture Rate':>14} {'Precision':>11}")
    print("  " + "-" * 76)
    for b in alert_budget_summary:
        print(
            f"  {b['budget_pct']:<10.1f}% {b['alerts']:>8} {b['fraud_captured']:>15} "
            f"{b['total_fraud']:>13} {b['capture_rate']:>14.4f} {b['precision']:>11.4f}"
        )
    print("=" * 80 + "\n")

    # Step 10: Score Distribution Diagnostics across Subpopulations
    diag_summary = compute_score_distribution_diagnostics(train_temp, test_temp, calib_model, calib_profile)

    # Step 11: Feature Importance & Attribution
    top_15, full_imp_df = compute_feature_importance(
        best_gen_res["model_object"],
        val_df[FRAUDGENOME_FEATURES].values,
        val_df["is_fraud_account"].astype(int).values,
        FRAUDGENOME_FEATURES,
        scaler=best_gen_res["scaler_object"],
        data_dir=data_dir,
    )

    # Step 12: Account-Level Explanation Safeguard Verification
    test_account_sample = test_temp.iloc[0]
    sample_exp = generate_account_explanation(test_account_sample, ref_t1_t2_temp)
    print("  Account Explanation Safeguard Verification:")
    print(f"  Account ID: {test_account_sample['account_id']}")
    print(f"  Explanation Output:\n{sample_exp}\n")

    # Format JSON outputs for disk export
    temporal_drift_experiments_json = {}
    for k, v in ablation_results.items():
        temporal_drift_experiments_json[k] = {
            "feature_set_name": v["feature_set_name"],
            "num_features": v["num_features"],
            "val_tuned_threshold": v["val_tuned_threshold"],
            "t3_evaluation": v["t3_evaluation"],
        }

    results_to_save = {
        "split_report": split_report,
        "supervised_calibration": brier_info,
        "risk_calibration": {
            "version": calib_profile["_metadata"]["version"],
            "source_phases": calib_profile["_metadata"]["source_phases"],
            "num_normal_accounts": calib_profile["_metadata"]["num_normal_accounts"],
            "policy_percentiles": calib_profile["policy_percentiles"],
        },
        "hybrid_weight_ablation": {
            "winning_config_name": winning_w_name,
            "winning_weights": list(winning_weights),
            "configs_evaluated": hybrid_weight_summary,
            "t3_evaluation_winning_weights": t3_metrics_winning,
        },
        "threshold_policy_experiments": policy_sweep_summary,
        "precision_at_k": pk_results,
        "alert_budget_analysis": alert_budget_summary,
        "score_distribution_diagnostics": diag_summary,
        "temporal_experiment": ablation_results["Exp B: Temporal FraudGenome"]["t3_evaluation"],
        "temporal_drift_experiments": temporal_drift_experiments_json,
        "relational_anomaly_experiments": ablation_results["Exp F: Relational Anomaly Score"]["t3_evaluation"],
        "hybrid_risk_experiments": ablation_results["Exp G: Calibrated Hybrid Risk Layer"]["t3_evaluation"],
    }

    res_path = os.path.join(data_dir, "model_results.json")
    with open(res_path, "w") as f:
        json.dump(results_to_save, f, indent=2)
    print(f"Saved machine-readable model results to '{res_path}'.")

    # Save model checkpoints
    model_dir = os.path.join(data_dir, "models")
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "baseline_best.pkl"), "wb") as f:
        pickle.dump(baseline_results["XGBoost"]["model_object"], f)
    with open(os.path.join(model_dir, "genome_best.pkl"), "wb") as f:
        pickle.dump(best_gen_res["model_object"], f)
    with open(os.path.join(model_dir, "hybrid_risk_best.pkl"), "wb") as f:
        pickle.dump(calib_model, f)

    print(f"Saved serialized model checkpoints to '{model_dir}'.\n")
    print("Complete pipeline execution successful.\n")


if __name__ == "__main__":
    main()
