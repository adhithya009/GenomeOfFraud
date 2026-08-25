"""
model.py — FraudGenome Model & Leakage-Safe Data Splitting Pipeline

Implements a TRUE Cluster-Grouped + Account-Grouped Stratified Data Splitting Strategy:
1. Primary Split: Cluster-Grouped + Account-Grouped Stratified Split (70/15/15)
   - Enforces 100% isolation of non-'none' `fraud_cluster_id` values across Train, Validation, and Test.
   - Grouping by `account_id` ensures all temporal rows (T1, T2, T3) of an account stay in the exact same split.
   - Eliminates both Account Identity Leakage and Fraud-Cluster Structural Leakage between splits.
   - Programmatically asserts `no_account_overlap` AND `no_fraud_cluster_overlap`.

2. Secondary Split: Temporal Experiment Split (T1+T2 -> T3) for out-of-time mutation evaluation.

IMPORTANT:
Excludes identifiers (`account_id`), target labels (`is_fraud_account`, `fraud_cluster_id`),
metadata (`phase`), and leaked features (`graph_community_fraud_ratio`).
"""

import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb


# ---------------------------------------------------------------------------
# Feature group definitions
# ---------------------------------------------------------------------------

# Columns that are identifiers, labels, metadata, or leaked — never used as predictive features
EXCLUDED_COLUMNS = [
    "account_id",
    "phase",
    "is_fraud_account",      # target label
    "fraud_cluster_id",      # target label/metadata
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

    # Verify all feature columns actually exist
    missing = [c for c in FRAUDGENOME_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Features missing from genome_full.csv: {missing}")


# ---------------------------------------------------------------------------
# 3. True Cluster-Grouped + Account-Grouped Data Splitting Pipeline
# ---------------------------------------------------------------------------

def prepare_cluster_aware_group_split(df, seed=42):
    """Perform a leakage-safe Cluster-Grouped + Account-Grouped Stratified Split (70/15/15).

    Methodology:
    1. Fraud Cluster Isolation:
       - Extracts unique non-'none' `fraud_cluster_id` values.
       - Deterministically allocates clusters so that no single `fraud_cluster_id` appears
         in more than one split (Train: ~60%, Val: ~20%, Test: ~20%).
       - All accounts belonging to a fraud cluster stay in the split assigned to that cluster.
    2. Legitimate Account Distribution:
       - Legitimate accounts (`fraud_cluster_id == 'none'`) are split 70% Train, 15% Validation, 15% Test.
    3. Account-Level Group Isolation:
       - All temporal phase rows (T1, T2, T3) for an account stay together in the same split.

    Args:
        df: Full genome DataFrame (6000 rows x 50 cols).
        seed: Random seed for reproducibility.

    Returns:
        train_df, val_df, test_df, split_report_dict
    """
    acct_info = df.groupby("account_id").agg(
        is_fraud=("is_fraud_account", "any"),
        cluster_id=("fraud_cluster_id", lambda x: [i for i in x if i != "none"][0] if any(i != "none" for i in x) else "none")
    ).reset_index()

    # Step 1: Identify all unique non-'none' fraud clusters
    fraud_clusters = sorted([c for c in acct_info["cluster_id"].unique() if c != "none"])

    # Allocate clusters: ~60% Train (3 clusters), ~20% Val (1 cluster), ~20% Test (1 cluster)
    n_clusters = len(fraud_clusters)

    # Use reproducible deterministic partition
    rng = np.random.RandomState(seed)
    shuffled_clusters = list(fraud_clusters)
    rng.shuffle(shuffled_clusters)

    n_train_c = max(int(np.round(0.60 * n_clusters)), 1)
    n_val_c = max(int(np.round(0.20 * n_clusters)), 1)

    train_clusters = sorted(shuffled_clusters[:n_train_c])
    val_clusters = sorted(shuffled_clusters[n_train_c:n_train_c + n_val_c])
    test_clusters = sorted(shuffled_clusters[n_train_c + n_val_c:])

    # Fraud accounts per split
    fraud_accts = acct_info[acct_info["is_fraud"]]
    train_fraud_accts = fraud_accts[fraud_accts["cluster_id"].isin(train_clusters)]
    val_fraud_accts = fraud_accts[fraud_accts["cluster_id"].isin(val_clusters)]
    test_fraud_accts = fraud_accts[fraud_accts["cluster_id"].isin(test_clusters)]

    # Step 2: Split legitimate accounts ('none') 70% Train, 15% Validation, 15% Test
    normal_accts = acct_info[~acct_info["is_fraud"]]
    train_norm, test_val_norm = train_test_split(
        normal_accts, test_size=0.30, random_state=seed
    )
    val_norm, test_norm = train_test_split(
        test_val_norm, test_size=0.50, random_state=seed
    )

    # Step 3: Combine account IDs per split
    train_acct_ids = set(train_fraud_accts["account_id"]).union(set(train_norm["account_id"]))
    val_acct_ids = set(val_fraud_accts["account_id"]).union(set(val_norm["account_id"]))
    test_acct_ids = set(test_fraud_accts["account_id"]).union(set(test_norm["account_id"]))

    # Step 4: Filter row-level DataFrame (all 3 phases for selected accounts)
    train_df = df[df["account_id"].isin(train_acct_ids)].copy().reset_index(drop=True)
    val_df = df[df["account_id"].isin(val_acct_ids)].copy().reset_index(drop=True)
    test_df = df[df["account_id"].isin(test_acct_ids)].copy().reset_index(drop=True)

    # Step 5: Programmatic validation assertions
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

    # 1. Account Overlap Checks
    tv_acct_overlap = len(train_accts.intersection(val_accts))
    tt_acct_overlap = len(train_accts.intersection(test_accts))
    vt_acct_overlap = len(val_accts.intersection(test_accts))

    no_account_overlap = (tv_acct_overlap == 0 and tt_acct_overlap == 0 and vt_acct_overlap == 0)
    assert no_account_overlap, (
        f"ACCOUNT LEAKAGE ASSERTION FAILED! Overlaps: Train-Val={tv_acct_overlap}, "
        f"Train-Test={tt_acct_overlap}, Val-Test={vt_acct_overlap}"
    )

    # 2. Fraud Cluster Overlap Checks (for non-'none' clusters)
    c_train = set(train_clusters)
    c_val = set(val_clusters)
    c_test = set(test_clusters)

    tv_cluster_overlap = c_train.intersection(c_val)
    tt_cluster_overlap = c_train.intersection(c_test)
    vt_cluster_overlap = c_val.intersection(c_test)

    no_fraud_cluster_overlap = (
        len(tv_cluster_overlap) == 0 and len(tt_cluster_overlap) == 0 and len(vt_cluster_overlap) == 0
    )
    assert no_fraud_cluster_overlap, (
        f"FRAUD CLUSTER LEAKAGE ASSERTION FAILED! Overlapping clusters: "
        f"Train-Val={tv_cluster_overlap}, Train-Test={tt_cluster_overlap}, Val-Test={vt_cluster_overlap}"
    )

    # 3. Overall Leakage-Safe Condition
    leakage_safe = no_account_overlap and no_fraud_cluster_overlap
    assert leakage_safe, "LEAKAGE SAFETY ASSERTION FAILED!"

    # 4. Sample and Fraud Metrics
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

    print("=" * 80)
    print("  TEMPORAL EXPERIMENT SPLIT (T1+T2 -> T3)")
    print("=" * 80)
    print(f"  Train Rows (T1+T2): {len(train_df)}  | Fraud: {train_df['is_fraud_account'].sum()} ({100*train_df['is_fraud_account'].mean():.2f}%)")
    print(f"  Test Rows (T3):     {len(test_df)}  | Fraud: {test_df['is_fraud_account'].sum()} ({100*test_df['is_fraud_account'].mean():.2f}%)")
    print("=" * 80 + "\n")

    return train_df, test_df, t2_ref_df


# ---------------------------------------------------------------------------
# 4. Pipeline Smoke Test
# ---------------------------------------------------------------------------

def smoke_test_split(train_df, val_df, test_df):
    """Minimal smoke test verifying that split feature matrices feed cleanly into XGBoost."""
    print("Running split pipeline smoke test...")

    X_train = train_df[FRAUDGENOME_FEATURES].values
    y_train = train_df["is_fraud_account"].astype(int).values

    X_val = val_df[FRAUDGENOME_FEATURES].values
    y_val = val_df["is_fraud_account"].astype(int).values

    X_test = test_df[FRAUDGENOME_FEATURES].values
    y_test = test_df["is_fraud_account"].astype(int).values

    n_pos = int(y_train.sum())
    scale_pos_weight = (len(y_train) - n_pos) / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=10,
        max_depth=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=42,
    )
    model.fit(X_train, y_train, verbose=False)

    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    print(f"  ✓ Smoke test passed! Train shape: {X_train.shape}, Val shape: {X_val.shape}, Test shape: {X_test.shape}")
    print(f"  ✓ Val predictions range: [{val_preds.min():.4f}, {val_preds.max():.4f}]")
    print(f"  ✓ Test predictions range: [{test_preds.min():.4f}, {test_preds.max():.4f}]\n")


# ---------------------------------------------------------------------------
# 5. Main Execution
# ---------------------------------------------------------------------------

def main():
    df, data_dir = load_features()

    # Step 1: Feature Audit & Leakage Inspection
    audit_features(df)

    # Step 2: Prepare Leakage-Safe True Cluster-Grouped + Account-Grouped Split
    train_df, val_df, test_df, split_report = prepare_cluster_aware_group_split(
        df, seed=42
    )

    # Step 3: Run Pipeline Smoke Test
    smoke_test_split(train_df, val_df, test_df)

    # Step 4: Save Split Metadata
    split_info_path = os.path.join(data_dir, "split_report.json")
    with open(split_info_path, "w") as f:
        json.dump(split_report, f, indent=2)
    print(f"Saved split metadata to '{split_info_path}'.\n")


if __name__ == "__main__":
    main()
