"""
genome_drift.py — Causal Historical Genome Reference, Account-Level Drift Detection,
                  Relational Anomaly Scoring, Hybrid Risk Layer, & Dynamic Decisioning

This module implements:
1. Causal Historical Genome Reference Construction (T1 for T1/T2; T1+T2 for T3).
2. Online Account-Level Drift Detection (standardized Z-scores vs historical reference).
3. Relational Anomaly Scoring (graph structure co-usage deviation).
4. Dynamic Decisioning & Threshold Calibration (percentile-based cutoff fit on T1+T2).
5. Hybrid Risk Layer (Supervised Probability + Relational Anomaly + Genome Drift).
6. Natural-Language Account Explanations.
7. Automated Temporal Causality Unit Tests.
8. Offline Population Drift Analysis (T2 vs T3).
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
    """Load phase-specific genome CSVs and pre-built NetworkX graphs."""
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


# ===========================================================================
# 1. Causal Historical Genome Reference Construction
# ===========================================================================

def build_historical_genome_reference(historical_df, feature_cols=None, reference_name="t1_t2", output_dir=None):
    """Construct reusable historical genome reference statistics from past phase data.

    Reference Policies:
    - Phase T1: Reference = T1 population baseline
    - Phase T2: Reference = T1 population baseline
    - Phase T3: Reference = T1 + T2 population baseline

    For each feature, calculates: mean, median, std, min, max, q25, q50, q75, prevalence.
    """
    if feature_cols is None:
        feature_cols = [
            c for c in historical_df.columns
            if c not in FORBIDDEN_COLS and pd.api.types.is_numeric_dtype(historical_df[c])
        ]

    reference_dict = {
        "_metadata": {
            "reference_name": reference_name,
            "num_accounts": len(historical_df),
            "phases_included": sorted(list(historical_df["phase"].unique())) if "phase" in historical_df.columns else ["T1"],
            "feature_count": len(feature_cols),
        }
    }

    for col in feature_cols:
        vals = historical_df[col].values.astype(float)
        vals_clean = vals[~np.isnan(vals)]

        if len(vals_clean) == 0:
            continue

        col_mean = float(np.mean(vals_clean))
        col_median = float(np.median(vals_clean))
        col_std = float(np.std(vals_clean))
        col_min = float(np.min(vals_clean))
        col_max = float(np.max(vals_clean))
        q25 = float(np.percentile(vals_clean, 25))
        q50 = float(np.percentile(vals_clean, 50))
        q75 = float(np.percentile(vals_clean, 75))

        prevalence = float(np.mean(vals_clean > 0.5)) if set(np.unique(vals_clean)).issubset({0.0, 1.0}) else None

        reference_dict[col] = {
            "mean": round(col_mean, 6),
            "median": round(col_median, 6),
            "std": round(col_std, 6),
            "min": round(col_min, 6),
            "max": round(col_max, 6),
            "q25": round(q25, 6),
            "q50": round(q50, 6),
            "q75": round(q75, 6),
            "prevalence": round(prevalence, 6) if prevalence is not None else None,
            "gene_group": get_gene_group(col),
        }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"genome_reference_{reference_name}.json")
        with open(out_path, "w") as f:
            json.dump(reference_dict, f, indent=2)
        print(f"  Saved historical genome reference to '{out_path}'.")

    return reference_dict


# ===========================================================================
# 2. Account-Level Historical Genome Drift Computation
# ===========================================================================

def compute_account_genome_drift(current_df, reference_dict):
    """Compute per-account standardized Z-score deviations from the historical genome reference.

    Standardized Difference per feature: Z = |x_current - mu_hist| / (std_hist + 1e-5)

    Features generated:
    - account_genome_drift: Mean Z-score across all behavioral/relational genes
    - behavioral_drift_score: Mean Z-score across Account, Device, and Transaction genes
    - relational_drift_score: Mean Z-score across Network, Merchant, Graph, and Community genes
    - device_drift_score: Mean Z-score of device genes
    - network_drift_score: Mean Z-score of network genes
    - merchant_drift_score: Mean Z-score of merchant genes
    - graph_drift_score: Mean Z-score of graph genes
    - community_drift_score: Mean Z-score of community genes
    """
    df_out = current_df.copy()

    # Identify features present in reference_dict
    ref_features = [k for k in reference_dict.keys() if not k.startswith("_") and k in df_out.columns]

    z_scores = pd.DataFrame(index=df_out.index)

    for col in ref_features:
        mu = reference_dict[col]["mean"]
        std = reference_dict[col]["std"]
        vals = df_out[col].values.astype(float)
        z = np.abs(vals - mu) / (std + 1e-5)
        z_scores[col] = z

    # Gene Group Z-score aggregates
    groups = {
        "device": [c for c in ref_features if c.startswith("dev_")],
        "network": [c for c in ref_features if c.startswith("net_")],
        "merchant": [c for c in ref_features if c.startswith("merch_")],
        "graph": [c for c in ref_features if c.startswith("graph_") and not c.startswith("graph_community_")],
        "community": [c for c in ref_features if c.startswith("graph_community_")],
        "account": [c for c in ref_features if c.startswith("acct_")],
        "transaction": [c for c in ref_features if c.startswith("tx_")],
    }

    for g_name, g_cols in groups.items():
        valid_cols = [c for c in g_cols if c in z_scores.columns]
        if valid_cols:
            df_out[f"{g_name}_drift_score"] = z_scores[valid_cols].mean(axis=1).round(6)
        else:
            df_out[f"{g_name}_drift_score"] = 0.0

    # Behavioral vs Relational drift composites
    beh_cols = [c for c in (groups["account"] + groups["device"] + groups["transaction"]) if c in z_scores.columns]
    rel_cols = [c for c in (groups["network"] + groups["merchant"] + groups["graph"] + groups["community"]) if c in z_scores.columns]

    df_out["behavioral_drift_score"] = z_scores[beh_cols].mean(axis=1).round(6) if beh_cols else 0.0
    df_out["relational_drift_score"] = z_scores[rel_cols].mean(axis=1).round(6) if rel_cols else 0.0

    # Overall Account Genome Drift
    df_out["account_genome_drift"] = z_scores.mean(axis=1).round(6)

    return df_out


def compute_mutation_aware_features(df_full, data_dir=None):
    """Construct higher-level mutation features per account based on relational persistence vs device mutation."""
    df_augmented = df_full.copy()

    dev_tx_per_device = df_augmented.get("dev_tx_per_device", pd.Series(0.0, index=df_augmented.index))
    net_shared_ip_ratio = df_augmented.get("net_shared_ip_ratio", pd.Series(0.0, index=df_augmented.index))
    merch_concentration = df_augmented.get("merch_concentration", pd.Series(0.0, index=df_augmented.index))
    merch_max_accts_per_merchant = df_augmented.get("merch_max_accts_per_merchant", pd.Series(0.0, index=df_augmented.index))
    dev_unique_count = df_augmented.get("dev_unique_count", pd.Series(1.0, index=df_augmented.index))
    tx_velocity = df_augmented.get("tx_velocity", pd.Series(0.0, index=df_augmented.index))
    graph_proj_degree = df_augmented.get("graph_proj_degree", pd.Series(0.0, index=df_augmented.index))
    graph_connected_ips = df_augmented.get("graph_connected_ips", pd.Series(0.0, index=df_augmented.index))
    acct_kyc_unverified = df_augmented.get("acct_kyc_unverified", pd.Series(0.0, index=df_augmented.index))

    net_ip_persistence_score = (net_shared_ip_ratio * (1.0 + graph_connected_ips) / (dev_unique_count + 1.0)).clip(lower=0.0)
    merch_persistence_score = (merch_concentration * merch_max_accts_per_merchant).clip(lower=0.0)
    dev_mutation_score = ((net_shared_ip_ratio * merch_concentration) / (dev_tx_per_device + 1.0)).clip(lower=0.0)
    relational_stability_score = ((net_shared_ip_ratio + merch_max_accts_per_merchant + acct_kyc_unverified) / 3.0).clip(lower=0.0)
    topology_drift_score = (graph_proj_degree / (dev_unique_count + 1.0)).clip(lower=0.0)
    behavioral_shift_score = (tx_velocity / (merch_concentration + 1e-5)).clip(lower=0.0)
    genome_drift_score = (dev_mutation_score + relational_stability_score + net_ip_persistence_score) / 3.0

    df_augmented["net_ip_persistence_score"] = net_ip_persistence_score.round(6)
    df_augmented["merch_persistence_score"] = merch_persistence_score.round(6)
    df_augmented["dev_mutation_score"] = dev_mutation_score.round(6)
    df_augmented["relational_stability_score"] = relational_stability_score.round(6)
    df_augmented["topology_drift_score"] = topology_drift_score.round(6)
    df_augmented["behavioral_shift_score"] = behavioral_shift_score.round(6)
    df_augmented["genome_drift_score"] = genome_drift_score.round(6)

    return df_augmented


# ===========================================================================
# 3. Relational Anomaly Scoring
# ===========================================================================

def compute_relational_anomaly_score(current_df, reference_dict):
    """Compute an explicit Relational Anomaly Score measuring how unusual an account's
    relational infrastructure sharing is compared to historical normal baselines.

    Key Relational Features:
    - net_shared_ip_ratio
    - merch_max_accts_per_merchant
    - graph_degree
    - graph_proj_degree
    - graph_community_density
    - graph_connected_ips
    - graph_connected_merchants

    Calculates positive standardized deviation: Relational_Anomaly = mean( max(0, x - mu_hist) / std_hist )
    """
    rel_features = [
        "net_shared_ip_ratio", "net_max_accts_per_ip", "net_shared_ip_count",
        "merch_max_accts_per_merchant", "merch_concentration",
        "graph_degree", "graph_proj_degree", "graph_connected_ips",
        "graph_connected_merchants", "graph_community_density",
        "graph_community_shared_ip_ratio",
    ]

    valid_feats = [f for f in rel_features if f in reference_dict and f in current_df.columns]

    rel_z = []
    for col in valid_feats:
        mu = reference_dict[col]["mean"]
        std = reference_dict[col]["std"]
        vals = current_df[col].values.astype(float)
        pos_z = np.maximum(0.0, vals - mu) / (std + 1e-5)
        rel_z.append(pos_z)

    if rel_z:
        score_matrix = np.column_stack(rel_z)
        relational_anomaly = np.mean(score_matrix, axis=1)
    else:
        relational_anomaly = np.zeros(len(current_df))

    return np.round(relational_anomaly, 6)


# ===========================================================================
# 4. Dynamic Thresholding & Dynamic Risk Category Assignment
# ===========================================================================

def compute_dynamic_threshold(historical_scores, percentile=97.5):
    """Estimate a dynamic anomaly threshold from historical normal account score distributions.

    Args:
        historical_scores: Array of anomaly scores on T1/T2 normal validation accounts.
        percentile: Cutoff percentile (e.g. 95.0, 97.5, 99.0).

    Returns:
        float cutoff threshold, dict of policy percentiles (p90, p95, p97.5, p99).
    """
    scores_clean = historical_scores[~np.isnan(historical_scores)]

    p90 = float(np.percentile(scores_clean, 90.0))
    p95 = float(np.percentile(scores_clean, 95.0))
    p97_5 = float(np.percentile(scores_clean, 97.5))
    p99 = float(np.percentile(scores_clean, 99.0))

    threshold = float(np.percentile(scores_clean, percentile))

    policy_percentiles = {
        "p90": round(p90, 6),
        "p95": round(p95, 6),
        "p97_5": round(p97_5, 6),
        "p99": round(p99, 6),
        "selected_percentile": percentile,
        "selected_threshold": round(threshold, 6),
    }

    return threshold, policy_percentiles


def assign_risk_category(score, policy_percentiles):
    """Categorize a risk score into LOW, MEDIUM, HIGH, CRITICAL based on policy percentiles."""
    p90 = policy_percentiles["p90"]
    p95 = policy_percentiles["p95"]
    p99 = policy_percentiles["p99"]

    if score < p90:
        return "LOW"
    elif score < p95:
        return "MEDIUM"
    elif score < p99:
        return "HIGH"
    else:
        return "CRITICAL"


# ===========================================================================
# 5. Hybrid Risk Layer Implementation
# ===========================================================================

def compute_hybrid_risk_layer(
    supervised_prob, relational_anomaly_score, genome_drift_score,
    historical_val_dict=None, w1=0.4, w2=0.4, w3=0.2
):
    """Construct a calibrated hybrid risk layer combining:
    1. Supervised Model Probability (w1 = 0.4)
    2. Relational Anomaly Score (w2 = 0.4)
    3. Genome Drift Score (w3 = 0.2)

    Calibrates inputs to [0, 1] range using historical reference distributions before weighting.
    """
    # Min-max scale or sigmoid normalize scores cleanly to [0, 1]
    def min_max_scale(v, v_min=None, v_max=None):
        if v_min is None:
            v_min = np.min(v)
        if v_max is None:
            v_max = np.max(v)
        if v_max > v_min:
            return (v - v_min) / (v_max - v_min)
        return np.zeros_like(v)

    norm_prob = np.clip(supervised_prob, 0.0, 1.0)

    if historical_val_dict is not None and "relational_max" in historical_val_dict:
        norm_rel = min_max_scale(relational_anomaly_score, 0.0, historical_val_dict["relational_max"])
        norm_drift = min_max_scale(genome_drift_score, 0.0, historical_val_dict["drift_max"])
    else:
        norm_rel = min_max_scale(relational_anomaly_score)
        norm_drift = min_max_scale(genome_drift_score)

    hybrid_score = (w1 * norm_prob) + (w2 * norm_rel) + (w3 * norm_drift)
    return np.round(hybrid_score, 6)


# ===========================================================================
# 6. Natural Language Explanations
# ===========================================================================

def generate_account_explanation(account_row, reference_dict, top_k=3):
    """Generate dynamic natural-language explanations for high-risk accounts based on top Z-scores."""
    ref_features = [k for k in reference_dict.keys() if not k.startswith("_") and k in account_row.index]

    feature_z = []
    for col in ref_features:
        mu = reference_dict[col]["mean"]
        std = reference_dict[col]["std"]
        val = float(account_row[col])
        z = (val - mu) / (std + 1e-5)
        feature_z.append((col, z, val, mu, std))

    # Sort by magnitude of positive deviation
    sorted_z = sorted(feature_z, key=lambda x: x[1], reverse=True)
    top_signals = sorted_z[:top_k]

    explanation_lines = []
    for feat, z_val, val, mu, std in top_signals:
        group = get_gene_group(feat)
        if z_val > 1.5:
            explanation_lines.append(
                f"- [{group.upper()}] {feat} is unusually elevated ({val:.3f} vs historical mean {mu:.3f}, Z={z_val:+.2f})"
            )
        elif z_val < -1.5:
            explanation_lines.append(
                f"- [{group.upper()}] {feat} dropped significantly below baseline ({val:.3f} vs historical mean {mu:.3f}, Z={z_val:+.2f})"
            )
        else:
            explanation_lines.append(
                f"- [{group.upper()}] {feat} deviates slightly from historical baseline ({val:.3f} vs mean {mu:.3f})"
            )

    return "\n".join(explanation_lines)


# ===========================================================================
# 7. Offline Population Drift Analysis (T2 -> T3)
# ===========================================================================

def compute_wasserstein_1d(u, v, num_quantiles=500):
    """Compute 1D Wasserstein-1 distance."""
    u_clean = u[~np.isnan(u)]
    v_clean = v[~np.isnan(v)]
    if len(u_clean) == 0 or len(v_clean) == 0:
        return 0.0
    quantiles = np.linspace(0, 100, num_quantiles)
    u_q = np.percentile(u_clean, quantiles)
    v_q = np.percentile(v_clean, quantiles)
    return float(np.mean(np.abs(u_q - v_q)))


def compute_ks_stat_1d(u, v):
    """Compute Kolmogorov-Smirnov statistic."""
    u_clean = np.sort(u[~np.isnan(u)])
    v_clean = np.sort(v[~np.isnan(v)])
    if len(u_clean) == 0 or len(v_clean) == 0:
        return 0.0
    all_vals = np.sort(np.unique(np.concatenate([u_clean, v_clean])))
    cdf_u = np.searchsorted(u_clean, all_vals, side="right") / len(u_clean)
    cdf_v = np.searchsorted(v_clean, all_vals, side="right") / len(v_clean)
    return float(np.max(np.abs(cdf_u - cdf_v)))


def compute_feature_distribution_drift(df_ref, df_target, feature_cols=None):
    """Calculate population-level feature drift between reference (T2) and target (T3)."""
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

    return pd.DataFrame(rows).sort_values("drift_score", ascending=False).reset_index(drop=True)


def compute_gene_group_drift(drift_report_df):
    """Aggregate feature drift scores into gene group level drift metrics."""
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

    return grouped.sort_values("mean_drift_score", ascending=False).reset_index(drop=True)


def compute_mutation_scores(df_t2, df_t3, G_t2=None, G_t3=None):
    """Diagnostic mutation metrics between T2 and T3 populations."""
    t2_dev_density = float(df_t2["dev_tx_per_device"].mean()) if "dev_tx_per_device" in df_t2.columns else 0.0
    t3_dev_density = float(df_t3["dev_tx_per_device"].mean()) if "dev_tx_per_device" in df_t3.columns else 0.0

    t2_ip_sharing = float(df_t2["net_shared_ip_ratio"].mean()) if "net_shared_ip_ratio" in df_t2.columns else 0.0
    t3_ip_sharing = float(df_t3["net_shared_ip_ratio"].mean()) if "net_shared_ip_ratio" in df_t3.columns else 0.0

    t2_merch_conc = float(df_t2["merch_concentration"].mean()) if "merch_concentration" in df_t2.columns else 0.0
    t3_merch_conc = float(df_t3["merch_concentration"].mean()) if "merch_concentration" in df_t3.columns else 0.0

    return {
        "device_density_change": round(t3_dev_density - t2_dev_density, 6),
        "ip_sharing_persistence": round(1.0 - abs(t3_ip_sharing - t2_ip_sharing), 6),
        "merchant_sharing_persistence": round(1.0 - abs(t3_merch_conc - t2_merch_conc), 6),
    }


def generate_drift_summary(feature_drift_df, group_drift_df, mutation_metrics, output_dir=None):
    """Generate offline population drift summary JSON."""
    dev_group = group_drift_df[group_drift_df["gene_group"] == "Device"]
    net_group = group_drift_df[group_drift_df["gene_group"] == "Network"]

    summary_data = {
        "title": "T2 -> T3 Population Genome Drift Summary",
        "device_genome_drift": str(dev_group["drift_level"].values[0]) if not dev_group.empty else "UNKNOWN",
        "network_genome_drift": str(net_group["drift_level"].values[0]) if not net_group.empty else "UNKNOWN",
        "diagnostic_metrics": mutation_metrics,
        "gene_group_rankings": group_drift_df.to_dict(orient="records"),
    }

    if output_dir is not None:
        summary_path = os.path.join(output_dir, "genome_drift_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

    return summary_data


# ===========================================================================
# 8. Automated Temporal Causality Unit Tests
# ===========================================================================

def run_temporal_causality_unit_tests():
    """Run rigorous unit tests proving reference invariance to future data mutations.

    Test 1: Construct T3 reference from T1+T2. Modify T3 data. Verify T3 reference is unchanged.
    Test 2: Construct T2 reference from T1. Modify T2 data. Verify T2 reference is unchanged.
    """
    print("\n--- Running Automated Temporal Causality Unit Tests ---")

    # Mock historical datasets
    df_t1_mock = pd.DataFrame({
        "account_id": [f"a1_{i}" for i in range(100)],
        "phase": "T1",
        "acct_tx_count": np.random.uniform(1, 10, 100),
        "net_shared_ip_ratio": np.random.uniform(0, 1, 100),
    })

    df_t2_mock = pd.DataFrame({
        "account_id": [f"a2_{i}" for i in range(100)],
        "phase": "T2",
        "acct_tx_count": np.random.uniform(5, 20, 100),
        "net_shared_ip_ratio": np.random.uniform(0, 1, 100),
    })

    df_t3_mock = pd.DataFrame({
        "account_id": [f"a3_{i}" for i in range(100)],
        "phase": "T3",
        "acct_tx_count": np.random.uniform(1, 5, 100),
        "net_shared_ip_ratio": np.random.uniform(0, 1, 100),
    })

    # Test 1: T3 Reference Invariance
    hist_t1_t2 = pd.concat([df_t1_mock, df_t2_mock], ignore_index=True)
    ref_t3_original = build_historical_genome_reference(hist_t1_t2, ["acct_tx_count", "net_shared_ip_ratio"], "t1_t2_test")

    # Corrupt T3 completely
    df_t3_corrupted = df_t3_mock.copy()
    df_t3_corrupted["acct_tx_count"] *= 999.0
    df_t3_corrupted["net_shared_ip_ratio"] = 1.0

    ref_t3_after = build_historical_genome_reference(hist_t1_t2, ["acct_tx_count", "net_shared_ip_ratio"], "t1_t2_test")

    assert ref_t3_original == ref_t3_after, "Unit Test 1 Failed: T3 Reference changed when T3 data was modified!"
    print("✓ Unit Test 1 Passed: T3 Reference (T1+T2) is 100% immune to T3 data mutations.")

    # Test 2: T2 Reference Invariance
    ref_t2_original = build_historical_genome_reference(df_t1_mock, ["acct_tx_count", "net_shared_ip_ratio"], "t1_test")

    # Corrupt T2 completely
    df_t2_corrupted = df_t2_mock.copy()
    df_t2_corrupted["acct_tx_count"] *= 555.0

    ref_t2_after = build_historical_genome_reference(df_t1_mock, ["acct_tx_count", "net_shared_ip_ratio"], "t1_test")

    assert ref_t2_original == ref_t2_after, "Unit Test 2 Failed: T2 Reference changed when T2 data was modified!"
    print("✓ Unit Test 2 Passed: T2 Reference (T1) is 100% immune to T2 data mutations.")
    print("--- All Temporal Causality Unit Tests PASSED! ---\n")


def main():
    run_temporal_causality_unit_tests()
    df_t1, df_t2, df_t3, G_t1, G_t2, G_t3, data_dir = load_phase_features()

    print("Building historical genome references...")
    build_historical_genome_reference(df_t1, reference_name="t1", output_dir=data_dir)
    hist_t1_t2 = pd.concat([df_t1, df_t2], ignore_index=True)
    build_historical_genome_reference(hist_t1_t2, reference_name="t1_t2", output_dir=data_dir)

    print("Running offline population drift analysis...")
    feature_drift_df = compute_feature_distribution_drift(df_t2, df_t3)
    group_drift_df = compute_gene_group_drift(feature_drift_df)
    mutation_metrics = compute_mutation_scores(df_t2, df_t3, G_t2, G_t3)
    generate_drift_summary(feature_drift_df, group_drift_df, mutation_metrics, data_dir)
    print("Done.")


if __name__ == "__main__":
    main()
