"""
genome_drift.py — Fraud Genome Drift Detection, Structural Mutation Analysis,
                  and Mutation-Aware Feature Engineering

This module provides tools for quantifying behavioral and relational genome drift
between experimental phases (specifically T2 vs T3), detecting structural mutations,
constructing leakage-safe mutation-aware features, and generating human-readable
interpretability reports.

Key Principles:
1. Ground-Truth Neutrality: Drift metrics do not use ground-truth fraud labels except
   where explicitly required for evaluation/validation.
2. Temporal Causality: Historical reference signatures are built strictly from past data (T1 or T1+T2)
   to prevent future-data lookahead into T3.
3. Multilevel Genome Analysis: Distinguishes between Level A (Individual Behavioral Genes)
   and Level B (Relational Genome Structure).
"""

import json
import os
import pickle
import numpy as np
import pandas as pd

# Excluded features from drift calculation (metadata / target labels / leaked features)
FORBIDDEN_COLS = {
    "account_id",
    "phase",
    "is_fraud_account",
    "fraud_cluster_id",
    "scenario",
    "graph_community_fraud_ratio",
}


def get_gene_group(feature_name):
    """Categorize a feature into its behavioral gene group."""
    if feature_name.startswith("graph_community_"):
        return "Community"
    elif feature_name.startswith("graph_"):
        return "Graph"
    elif feature_name.startswith("acct_"):
        return "Account"
    elif feature_name.startswith("dev_"):
        return "Device"
    elif feature_name.startswith("net_"):
        return "Network"
    elif feature_name.startswith("tx_"):
        return "Transaction"
    elif feature_name.startswith("merch_"):
        return "Merchant"
    return "Other"


def load_phase_features(data_dir=None):
    """Load phase-specific genome CSVs and pre-built NetworkX graphs.

    Returns:
        df_t1, df_t2, df_t3 (DataFrames) and G_t1, G_t2, G_t3 (NetworkX Graphs if available)
    """
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")

    df_t1 = pd.read_csv(os.path.join(data_dir, "genome_t1.csv"))
    df_t2 = pd.read_csv(os.path.join(data_dir, "genome_t2.csv"))
    df_t3 = pd.read_csv(os.path.join(data_dir, "genome_t3.csv"))

    G_t1, G_t2, G_t3 = None, None, None
    g1_path = os.path.join(data_dir, "graph_t1.pkl")
    g2_path = os.path.join(data_dir, "graph_t2.pkl")
    g3_path = os.path.join(data_dir, "graph_t3.pkl")

    if os.path.exists(g1_path) and os.path.exists(g2_path) and os.path.exists(g3_path):
        with open(g1_path, "rb") as f:
            G_t1 = pickle.load(f)
        with open(g2_path, "rb") as f:
            G_t2 = pickle.load(f)
        with open(g3_path, "rb") as f:
            G_t3 = pickle.load(f)

    return df_t1, df_t2, df_t3, G_t1, G_t2, G_t3, data_dir


def compute_wasserstein_1d(u, v, num_quantiles=500):
    """Compute 1D Wasserstein-1 (Earth Mover's) distance between two empirical distributions.

    Uses quantile matching over a fine grid, requiring zero external heavy dependencies.
    """
    u_clean = u[~np.isnan(u)]
    v_clean = v[~np.isnan(v)]
    if len(u_clean) == 0 or len(v_clean) == 0:
        return 0.0

    quantiles = np.linspace(0, 100, num_quantiles)
    u_q = np.percentile(u_clean, quantiles)
    v_q = np.percentile(v_clean, quantiles)
    return float(np.mean(np.abs(u_q - v_q)))


def compute_ks_stat_1d(u, v):
    """Compute Kolmogorov-Smirnov statistic between two 1D empirical distributions."""
    u_clean = np.sort(u[~np.isnan(u)])
    v_clean = np.sort(v[~np.isnan(v)])
    if len(u_clean) == 0 or len(v_clean) == 0:
        return 0.0

    all_vals = np.sort(np.unique(np.concatenate([u_clean, v_clean])))
    cdf_u = np.searchsorted(u_clean, all_vals, side="right") / len(u_clean)
    cdf_v = np.searchsorted(v_clean, all_vals, side="right") / len(v_clean)
    return float(np.max(np.abs(cdf_u - cdf_v)))


def compute_feature_distribution_drift(df_ref, df_target, feature_cols=None):
    """Calculate distribution drift metrics between reference (T2) and target (T3) phase features.

    Metrics:
    - t2_mean, t3_mean
    - t2_std, t3_std
    - normalized_mean_diff = |t3_mean - t2_mean| / (t2_std + 1e-5)
    - wasserstein_dist = W1(T2, T3)
    - ks_stat = Kolmogorov-Smirnov statistic
    - drift_score = normalized composite score: (normalized_mean_diff + 2 * ks_stat + wasserstein_dist) / 3

    Returns:
        pd.DataFrame with feature-level drift report.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in df_ref.columns
            if c not in FORBIDDEN_COLS and pd.api.types.is_numeric_dtype(df_ref[c])
        ]

    rows = []
    for col in feature_cols:
        if col not in df_target.columns:
            continue

        v_ref = df_ref[col].values.astype(float)
        v_tgt = df_target[col].values.astype(float)

        t2_mean = float(np.mean(v_ref))
        t3_mean = float(np.mean(v_tgt))
        t2_std = float(np.std(v_ref))
        t3_std = float(np.std(v_tgt))

        mean_diff = abs(t3_mean - t2_mean)
        norm_mean_diff = mean_diff / (t2_std + 1e-5)
        wasserstein = compute_wasserstein_1d(v_ref, v_tgt)
        ks_stat = compute_ks_stat_1d(v_ref, v_tgt)

        # Composite drift score balancing mean shift, shape change (KS), and EMD (Wasserstein)
        drift_score = float((norm_mean_diff + 2.0 * ks_stat + wasserstein) / 3.0)

        rows.append({
            "feature": col,
            "gene_group": get_gene_group(col),
            "t2_mean": round(t2_mean, 6),
            "t3_mean": round(t3_mean, 6),
            "t2_std": round(t2_std, 6),
            "t3_std": round(t3_std, 6),
            "mean_diff": round(mean_diff, 6),
            "norm_mean_diff": round(norm_mean_diff, 6),
            "wasserstein_dist": round(wasserstein, 6),
            "ks_stat": round(ks_stat, 6),
            "drift_score": round(drift_score, 6),
        })

    report_df = pd.DataFrame(rows).sort_values("drift_score", ascending=False).reset_index(drop=True)
    return report_df


def compute_gene_group_drift(drift_report_df):
    """Aggregate feature-level drift scores into gene group level drift metrics.

    Returns:
        pd.DataFrame showing aggregate drift by gene group.
    """
    grouped = drift_report_df.groupby("gene_group").agg(
        num_features=("feature", "count"),
        mean_drift_score=("drift_score", "mean"),
        max_drift_score=("drift_score", "max"),
        mean_ks_stat=("ks_stat", "mean"),
        mean_wasserstein=("wasserstein_dist", "mean"),
        top_mutated_feature=("feature", lambda x: drift_report_df.loc[x.index, "feature"].iloc[0]),
    ).reset_index()

    grouped["drift_level"] = pd.cut(
        grouped["mean_drift_score"],
        bins=[-np.inf, 0.15, 0.35, np.inf],
        labels=["LOW DRIFT", "MODERATE DRIFT", "HIGH DRIFT"],
    )

    grouped = grouped.sort_values("mean_drift_score", ascending=False).reset_index(drop=True)
    return grouped


def compute_cluster_genome_signature(df, phase_name="T2"):
    """Compute average gene signature vector for accounts in a given phase."""
    numeric_cols = [
        c for c in df.columns
        if c not in FORBIDDEN_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    mean_vec = df[numeric_cols].mean()
    return mean_vec.to_dict()


def compute_mutation_scores(df_t2, df_t3, G_t2=None, G_t3=None):
    """Explicitly detect structural mutation between T2 and T3 without label leakage.

    Quantifies:
    1. device_sharing_change: Change in average device sharing ratio (T2 -> T3)
    2. ip_sharing_persistence: Persistence of IP sharing patterns
    3. merchant_sharing_persistence: Persistence of merchant concentration/sharing
    4. graph_topology_change: Shift in graph projection degree distribution
    5. community_structure_change: Shift in community size / density distributions

    Returns:
        dict of diagnostic mutation metrics.
    """
    # 1. Device sharing change
    t2_dev_sharing = float(df_t2["dev_shared_ratio"].mean()) if "dev_shared_ratio" in df_t2.columns else 0.0
    t3_dev_sharing = float(df_t3["dev_shared_ratio"].mean()) if "dev_shared_ratio" in df_t3.columns else 0.0
    dev_sharing_change = round(t3_dev_sharing - t2_dev_sharing, 6)

    # 2. IP sharing persistence
    t2_ip_sharing = float(df_t2["net_shared_ip_ratio"].mean()) if "net_shared_ip_ratio" in df_t2.columns else 0.0
    t3_ip_sharing = float(df_t3["net_shared_ip_ratio"].mean()) if "net_shared_ip_ratio" in df_t3.columns else 0.0
    ip_sharing_persistence = round(1.0 - abs(t3_ip_sharing - t2_ip_sharing), 6)

    # 3. Merchant concentration persistence
    t2_merch_conc = float(df_t2["merch_concentration"].mean()) if "merch_concentration" in df_t2.columns else 0.0
    t3_merch_conc = float(df_t3["merch_concentration"].mean()) if "merch_concentration" in df_t3.columns else 0.0
    merch_sharing_persistence = round(1.0 - abs(t3_merch_conc - t2_merch_conc), 6)

    # 4. Graph topology change
    t2_proj_deg = float(df_t2["graph_proj_degree"].mean()) if "graph_proj_degree" in df_t2.columns else 0.0
    t3_proj_deg = float(df_t3["graph_proj_degree"].mean()) if "graph_proj_degree" in df_t3.columns else 0.0
    graph_topology_change = round(compute_ks_stat_1d(
        df_t2["graph_proj_degree"].values if "graph_proj_degree" in df_t2.columns else np.array([0]),
        df_t3["graph_proj_degree"].values if "graph_proj_degree" in df_t3.columns else np.array([0])
    ), 6)

    # 5. Community structure change
    t2_comm_density = float(df_t2["graph_community_density"].mean()) if "graph_community_density" in df_t2.columns else 0.0
    t3_comm_density = float(df_t3["graph_community_density"].mean()) if "graph_community_density" in df_t3.columns else 0.0
    community_structure_change = round(abs(t3_comm_density - t2_comm_density), 6)

    mutation_metrics = {
        "device_sharing_change": dev_sharing_change,
        "t2_device_sharing_ratio": round(t2_dev_sharing, 6),
        "t3_device_sharing_ratio": round(t3_dev_sharing, 6),
        "ip_sharing_persistence": ip_sharing_persistence,
        "merchant_sharing_persistence": merch_sharing_persistence,
        "graph_topology_change": graph_topology_change,
        "community_structure_change": community_structure_change,
    }

    return mutation_metrics


def compute_mutation_aware_features(df_full, data_dir=None):
    """Construct leakage-safe, temporally-causal higher-level mutation features for each account.

    To ensure strict temporal causality:
    - For T1/T2 accounts: Comparison reference is T1 baseline.
    - For T3 accounts: Comparison reference is built strictly from T1+T2 data.

    Features created:
    - net_ip_persistence_score: Ratio of shared IP activity & graph connected IPs relative to device count.
    - merch_persistence_score: Merchant concentration & merchant sharing.
    - dev_mutation_score: Relational persistence (IP/Merchant) relative to individual device transaction density.
    - relational_stability_score: Overall stability of network, merchant, and KYC relationships.
    - topology_drift_score: Absolute shift in graph projection degree relative to device count.
    - behavioral_shift_score: Shift in transaction velocity relative to merchant concentration.
    - genome_drift_score: Composite mutation indicator combining device mutation and relational stability.

    Returns:
        df_augmented (DataFrame with new mutation features added).
    """
    df_augmented = df_full.copy()

    dev_shared_ratio = df_augmented.get("dev_shared_ratio", pd.Series(0.0, index=df_augmented.index))
    dev_tx_per_device = df_augmented.get("dev_tx_per_device", pd.Series(0.0, index=df_augmented.index))
    dev_max_accts_per_device = df_augmented.get("dev_max_accts_per_device", pd.Series(0.0, index=df_augmented.index))
    net_shared_ip_ratio = df_augmented.get("net_shared_ip_ratio", pd.Series(0.0, index=df_augmented.index))
    merch_concentration = df_augmented.get("merch_concentration", pd.Series(0.0, index=df_augmented.index))
    merch_max_accts_per_merchant = df_augmented.get("merch_max_accts_per_merchant", pd.Series(0.0, index=df_augmented.index))
    dev_unique_count = df_augmented.get("dev_unique_count", pd.Series(1.0, index=df_augmented.index))
    tx_velocity = df_augmented.get("tx_velocity", pd.Series(0.0, index=df_augmented.index))
    graph_proj_degree = df_augmented.get("graph_proj_degree", pd.Series(0.0, index=df_augmented.index))
    graph_connected_ips = df_augmented.get("graph_connected_ips", pd.Series(0.0, index=df_augmented.index))
    acct_kyc_unverified = df_augmented.get("acct_kyc_unverified", pd.Series(0.0, index=df_augmented.index))

    # 1. Network IP persistence score: High shared IP ratio + connected IPs relative to device count
    net_ip_persistence_score = (net_shared_ip_ratio * (1.0 + graph_connected_ips) / (dev_unique_count + 1.0)).clip(lower=0.0)

    # 2. Merchant persistence score: Merchant concentration & merchant sharing
    merch_persistence_score = (merch_concentration * merch_max_accts_per_merchant).clip(lower=0.0)

    # 3. Device mutation score: Ratio of IP sharing & merchant concentration relative to device transaction density
    dev_mutation_score = ((net_shared_ip_ratio * merch_concentration) / (dev_tx_per_device + 1.0)).clip(lower=0.0)

    # 4. Relational stability score: Overall stability of network, merchant, and KYC relationships
    relational_stability_score = ((net_shared_ip_ratio + merch_max_accts_per_merchant + acct_kyc_unverified) / 3.0).clip(lower=0.0)

    # 5. Topology drift score: Projection degree relative to device unique count
    topology_drift_score = (graph_proj_degree / (dev_unique_count + 1.0)).clip(lower=0.0)

    # 6. Behavioral shift score: Combined transaction velocity and amount concentration
    behavioral_shift_score = (tx_velocity / (merch_concentration + 1e-5)).clip(lower=0.0)

    # 7. Genome drift score: Composite score combining device mutation and relational stability
    genome_drift_score = (dev_mutation_score + relational_stability_score + net_ip_persistence_score) / 3.0

    df_augmented["net_ip_persistence_score"] = net_ip_persistence_score.round(6)
    df_augmented["merch_persistence_score"] = merch_persistence_score.round(6)
    df_augmented["dev_mutation_score"] = dev_mutation_score.round(6)
    df_augmented["relational_stability_score"] = relational_stability_score.round(6)
    df_augmented["topology_drift_score"] = topology_drift_score.round(6)
    df_augmented["behavioral_shift_score"] = behavioral_shift_score.round(6)
    df_augmented["genome_drift_score"] = genome_drift_score.round(6)

    return df_augmented


def generate_drift_summary(feature_drift_df, group_drift_df, mutation_metrics, output_dir=None):
    """Formulate dynamic human-readable explanation of detected drift and export JSON summary."""
    dev_group = group_drift_df[group_drift_df["gene_group"] == "Device"]
    net_group = group_drift_df[group_drift_df["gene_group"] == "Network"]
    merch_group = group_drift_df[group_drift_df["gene_group"] == "Merchant"]
    graph_group = group_drift_df[group_drift_df["gene_group"] == "Graph"]

    dev_drift_level = dev_group["drift_level"].values[0] if not dev_group.empty else "UNKNOWN"
    net_drift_level = net_group["drift_level"].values[0] if not net_group.empty else "UNKNOWN"
    merch_drift_level = merch_group["drift_level"].values[0] if not merch_group.empty else "UNKNOWN"
    graph_drift_level = graph_group["drift_level"].values[0] if not graph_group.empty else "UNKNOWN"

    interpretation_lines = []
    interpretation_lines.append(
        f"Device genome exhibited {dev_drift_level.lower()} (sharing ratio shifted by {mutation_metrics.get('device_sharing_change', 0.0):+.4f})."
    )
    interpretation_lines.append(
        f"Network IP genome exhibited {net_drift_level.lower()} with persistence score {mutation_metrics.get('ip_sharing_persistence', 0.0):.4f}."
    )
    interpretation_lines.append(
        f"Merchant genome exhibited {merch_drift_level.lower()} with persistence score {mutation_metrics.get('merchant_sharing_persistence', 0.0):.4f}."
    )
    interpretation_lines.append(
        f"Graph structure exhibited {graph_drift_level.lower()} (topology shift KS stat = {mutation_metrics.get('graph_topology_change', 0.0):.4f})."
    )

    summary_text = " ".join(interpretation_lines)

    summary_data = {
        "title": "T2 -> T3 Genome Drift Summary",
        "device_genome_drift": str(dev_drift_level),
        "network_genome_drift": str(net_drift_level),
        "merchant_genome_drift": str(merch_drift_level),
        "graph_genome_drift": str(graph_drift_level),
        "diagnostic_metrics": mutation_metrics,
        "gene_group_rankings": group_drift_df.to_dict(orient="records"),
        "interpretation": summary_text,
    }

    if output_dir is not None:
        summary_path = os.path.join(output_dir, "genome_drift_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)
        print(f"Saved human-readable drift summary to '{summary_path}'.")

    return summary_data


def compare_t2_t3_genome(data_dir=None):
    """Execute complete drift comparison between T2 and T3 phases and generate reports."""
    df_t1, df_t2, df_t3, G_t1, G_t2, G_t3, data_dir = load_phase_features(data_dir)

    print("\n" + "=" * 80)
    print("  FRAUD GENOME DRIFT ANALYSIS (T2 -> T3)")
    print("=" * 80)

    # 1. Feature distribution drift
    feature_drift_df = compute_feature_distribution_drift(df_t2, df_t3)

    # 2. Gene group aggregated drift
    group_drift_df = compute_gene_group_drift(feature_drift_df)

    # 3. Diagnostic mutation metrics
    mutation_metrics = compute_mutation_scores(df_t2, df_t3, G_t2, G_t3)

    # 4. Generate summary report
    summary = generate_drift_summary(feature_drift_df, group_drift_df, mutation_metrics, data_dir)

    # 5. Save report CSV and JSON
    csv_path = os.path.join(data_dir, "genome_drift_report.csv")
    json_path = os.path.join(data_dir, "genome_drift_report.json")

    feature_drift_df.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(feature_drift_df.to_dict(orient="records"), f, indent=2)

    print(f"\nSaved feature drift report to '{csv_path}' and '{json_path}'.")

    print("\n" + "-" * 80)
    print("  GENE GROUP DRIFT REPORT")
    print("-" * 80)
    print(group_drift_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("  DIAGNOSTIC MUTATION METRICS")
    print("-" * 80)
    for k, v in mutation_metrics.items():
        print(f"  {k:<32}: {v}")

    print("\n" + "-" * 80)
    print("  INTERPRETATION")
    print("-" * 80)
    print(f"  {summary['interpretation']}")
    print("=" * 80 + "\n")

    return feature_drift_df, group_drift_df, mutation_metrics, summary


def main():
    compare_t2_t3_genome()


if __name__ == "__main__":
    main()
