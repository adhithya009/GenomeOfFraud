import os

import numpy as np
import pandas as pd
from genome_drift import compute_mutation_aware_features


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATEGORY_MAP = {"digital": 0, "food": 1, "retail": 2, "travel": 3}

# Binary / categorical features excluded from min-max normalization
NON_SCALED_COLS = {"acct_kyc_unverified", "merch_top_category"}

GENE_PREFIXES = [
    "acct_", "dev_", "net_", "tx_", "merch_", "graph_",
    "relational_", "topology_", "behavioral_", "genome_",
]



# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_all_inputs(data_dir=None):
    """Load raw CSVs and all pre-computed graph/community feature files."""
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")

    print(f"Loading inputs from '{data_dir}'...")

    transactions_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    labels_df = pd.read_csv(os.path.join(data_dir, "fraud_labels.csv"))
    accounts_df = pd.read_csv(os.path.join(data_dir, "accounts.csv"))
    merchants_df = pd.read_csv(os.path.join(data_dir, "merchants.csv"))

    merged_df = pd.merge(transactions_df, labels_df, on="transaction_id")
    merged_df["timestamp_dt"] = pd.to_datetime(merged_df["timestamp"])

    accounts_df["created_at_dt"] = pd.to_datetime(accounts_df["created_at"])
    merchants_df["created_at_dt"] = pd.to_datetime(merchants_df["created_at"])

    graph_features = {}
    community_stats = {}
    for phase in ("t1", "t2", "t3"):
        graph_features[phase.upper()] = pd.read_csv(
            os.path.join(data_dir, f"account_features_{phase}.csv")
        )
        community_stats[phase.upper()] = pd.read_csv(
            os.path.join(data_dir, f"community_stats_{phase}_louvain.csv")
        )

    return merged_df, accounts_df, merchants_df, graph_features, community_stats, data_dir


# ---------------------------------------------------------------------------
# 2. Phase Windows
# ---------------------------------------------------------------------------

def determine_phase_windows(merged_df):
    """Return start/end timestamps for T1, T2, T3 (same logic as build_graph.py)."""
    t_min = merged_df["timestamp_dt"].min()
    t_max = merged_df["timestamp_dt"].max()
    d = (t_max - t_min) / 3
    return {
        "T1": (t_min, t_min + d),
        "T2": (t_min + d, t_min + 2 * d),
        "T3": (t_min + 2 * d, t_max),
    }


# ---------------------------------------------------------------------------
# 3a. Gene Group — Account
# ---------------------------------------------------------------------------

def compute_account_genes(phase_df, accounts_df, all_account_ids, phase_start, phase_days):
    """Account metadata + phase transaction activity features."""
    phase_days = max(phase_days, 1)

    agg = phase_df.groupby("account_id").agg(
        acct_tx_count=("transaction_id", "count"),
        acct_amount_total=("amount", "sum"),
        acct_amount_avg=("amount", "mean"),
        acct_amount_std=("amount", "std"),
        acct_active_days=("timestamp_dt", lambda x: x.dt.date.nunique()),
    ).reset_index()

    base = pd.DataFrame({"account_id": all_account_ids})
    df = base.merge(agg, on="account_id", how="left").fillna(0).set_index("account_id")
    df["acct_tx_velocity"] = df["acct_tx_count"] / phase_days

    meta = accounts_df[["account_id", "created_at_dt", "kyc_status"]].copy()
    meta["acct_age_days"] = (phase_start - meta["created_at_dt"]).dt.days.clip(lower=0)
    meta["acct_kyc_unverified"] = (meta["kyc_status"] == "unverified").astype(int)
    meta = meta.set_index("account_id")

    df = df.join(meta[["acct_age_days", "acct_kyc_unverified"]])

    return df[[
        "acct_tx_count", "acct_tx_velocity", "acct_active_days",
        "acct_amount_total", "acct_amount_avg", "acct_amount_std",
        "acct_age_days", "acct_kyc_unverified",
    ]]


# ---------------------------------------------------------------------------
# 3b. Gene Group — Device
# ---------------------------------------------------------------------------

def compute_device_genes(phase_df, all_account_ids):
    """Device usage patterns: sharing, reuse, concentration."""
    # How many accounts used each device in this phase?
    device_acct_count = (
        phase_df.groupby("device_id")["account_id"].nunique()
        .rename("device_acct_count")
    )
    shared_device_set = set(device_acct_count[device_acct_count >= 2].index)

    enriched = phase_df.join(device_acct_count, on="device_id")

    per_acct = enriched.groupby("account_id").agg(
        _tx_count=("transaction_id", "count"),
        dev_unique_count=("device_id", "nunique"),
        dev_max_accts_per_device=("device_acct_count", "max"),
    )

    # Most-used device concentration
    top_dev = (
        phase_df.groupby(["account_id", "device_id"]).size()
        .reset_index(name="n")
    )
    top_dev = top_dev.loc[top_dev.groupby("account_id")["n"].idxmax()]
    top_dev = top_dev.set_index("account_id")["n"].rename("_top_dev_n")
    per_acct = per_acct.join(top_dev)
    per_acct["dev_concentration"] = per_acct["_top_dev_n"] / per_acct["_tx_count"].clip(lower=1)
    per_acct["dev_tx_per_device"] = per_acct["_tx_count"] / per_acct["dev_unique_count"].clip(lower=1)

    # Shared device count (unique shared devices per account)
    shared_rows = phase_df[phase_df["device_id"].isin(shared_device_set)]
    dev_shared = (
        shared_rows.groupby("account_id")["device_id"].nunique()
        .rename("dev_shared_count")
    )
    per_acct = per_acct.join(dev_shared).fillna({"dev_shared_count": 0})
    per_acct["dev_shared_ratio"] = (
        per_acct["dev_shared_count"] / per_acct["dev_unique_count"].clip(lower=1)
    )

    base = pd.DataFrame({"account_id": all_account_ids}).set_index("account_id")
    result = base.join(per_acct).fillna(0)

    return result[[
        "dev_unique_count", "dev_tx_per_device", "dev_max_accts_per_device",
        "dev_shared_count", "dev_shared_ratio", "dev_concentration",
    ]]


# ---------------------------------------------------------------------------
# 3c. Gene Group — Network (IP)
# ---------------------------------------------------------------------------

def compute_network_genes(phase_df, all_account_ids):
    """IP usage patterns: sharing, reuse, concentration."""
    ip_acct_count = (
        phase_df.groupby("ip_id")["account_id"].nunique()
        .rename("ip_acct_count")
    )
    shared_ip_set = set(ip_acct_count[ip_acct_count >= 2].index)

    enriched = phase_df.join(ip_acct_count, on="ip_id")

    per_acct = enriched.groupby("account_id").agg(
        _tx_count=("transaction_id", "count"),
        net_unique_ip_count=("ip_id", "nunique"),
        net_max_accts_per_ip=("ip_acct_count", "max"),
    )

    # Most-used IP concentration
    top_ip = (
        phase_df.groupby(["account_id", "ip_id"]).size()
        .reset_index(name="n")
    )
    top_ip = top_ip.loc[top_ip.groupby("account_id")["n"].idxmax()]
    top_ip = top_ip.set_index("account_id")["n"].rename("_top_ip_n")
    per_acct = per_acct.join(top_ip)
    per_acct["net_ip_concentration"] = per_acct["_top_ip_n"] / per_acct["_tx_count"].clip(lower=1)
    per_acct["net_tx_per_ip"] = per_acct["_tx_count"] / per_acct["net_unique_ip_count"].clip(lower=1)

    shared_rows = phase_df[phase_df["ip_id"].isin(shared_ip_set)]
    ip_shared = (
        shared_rows.groupby("account_id")["ip_id"].nunique()
        .rename("net_shared_ip_count")
    )
    per_acct = per_acct.join(ip_shared).fillna({"net_shared_ip_count": 0})
    per_acct["net_shared_ip_ratio"] = (
        per_acct["net_shared_ip_count"] / per_acct["net_unique_ip_count"].clip(lower=1)
    )

    base = pd.DataFrame({"account_id": all_account_ids}).set_index("account_id")
    result = base.join(per_acct).fillna(0)

    return result[[
        "net_unique_ip_count", "net_tx_per_ip", "net_max_accts_per_ip",
        "net_shared_ip_count", "net_shared_ip_ratio", "net_ip_concentration",
    ]]


# ---------------------------------------------------------------------------
# 3d. Gene Group — Transaction
# ---------------------------------------------------------------------------

def compute_transaction_genes(phase_df, all_account_ids, phase_days):
    """Transaction timing, velocity, and amount distribution features."""
    phase_days = max(phase_days, 1)

    # Time spread: hours between first and last transaction
    time_spread = phase_df.groupby("account_id")["timestamp_dt"].agg(["min", "max"])
    time_spread["tx_time_spread_hours"] = (
        (time_spread["max"] - time_spread["min"]).dt.total_seconds() / 3600
    )

    # Mode transaction hour
    phase_df = phase_df.copy()
    phase_df["_hour"] = phase_df["timestamp_dt"].dt.hour
    hour_mode = (
        phase_df.groupby("account_id")["_hour"]
        .agg(lambda x: x.mode().iloc[0])
        .rename("tx_hour_mode")
    )

    # Weekend ratio
    phase_df["_weekend"] = (phase_df["timestamp_dt"].dt.dayofweek >= 5).astype(int)
    weekend_ratio = phase_df.groupby("account_id")["_weekend"].mean().rename("tx_weekend_ratio")

    # Amount aggregates
    amt = phase_df.groupby("account_id")["amount"].agg(
        tx_amount_mean="mean",
        tx_amount_max="max",
        _total="sum",
        _count="count",
    )
    amt["tx_amount_concentration"] = amt["tx_amount_max"] / amt["_total"].clip(lower=0.01)
    amt["tx_velocity"] = amt["_count"] / phase_days

    per_acct = amt.join([time_spread[["tx_time_spread_hours"]], hour_mode, weekend_ratio])

    base = pd.DataFrame({"account_id": all_account_ids}).set_index("account_id")
    result = base.join(per_acct).fillna(0)

    return result[[
        "tx_velocity", "tx_amount_mean", "tx_amount_max",
        "tx_amount_concentration", "tx_time_spread_hours",
        "tx_hour_mode", "tx_weekend_ratio",
    ]]


# ---------------------------------------------------------------------------
# 3e. Gene Group — Merchant
# ---------------------------------------------------------------------------

def compute_merchant_genes(phase_df, merchants_df, all_account_ids, phase_start):
    """Merchant usage patterns: concentration, category, sharing, age."""
    merch_acct_count = (
        phase_df.groupby("merchant_id")["account_id"].nunique()
        .rename("merch_acct_count")
    )

    enriched = (
        phase_df
        .join(merch_acct_count, on="merchant_id")
        .merge(merchants_df[["merchant_id", "created_at_dt", "category"]], on="merchant_id", how="left")
    )
    enriched["merch_age_days"] = (phase_start - enriched["created_at_dt"]).dt.days.clip(lower=0)
    enriched["_cat_code"] = enriched["category"].map(CATEGORY_MAP).fillna(0).astype(int)

    per_acct = enriched.groupby("account_id").agg(
        _tx_count=("transaction_id", "count"),
        merch_unique_count=("merchant_id", "nunique"),
        merch_max_accts_per_merchant=("merch_acct_count", "max"),
        merch_avg_age_days=("merch_age_days", "mean"),
    )

    # Concentration: fraction of transactions at single most-used merchant
    top_merch = (
        phase_df.groupby(["account_id", "merchant_id"]).size()
        .reset_index(name="n")
    )
    top_merch = top_merch.loc[top_merch.groupby("account_id")["n"].idxmax()]
    top_merch = top_merch.set_index("account_id")["n"].rename("_top_merch_n")
    per_acct = per_acct.join(top_merch)
    per_acct["merch_concentration"] = per_acct["_top_merch_n"] / per_acct["_tx_count"].clip(lower=1)

    # Most frequent category per account
    top_cat = (
        enriched.groupby(["account_id", "_cat_code"]).size()
        .reset_index(name="n")
    )
    top_cat = top_cat.loc[top_cat.groupby("account_id")["n"].idxmax()]
    top_cat = top_cat.set_index("account_id")["_cat_code"].rename("merch_top_category")
    per_acct = per_acct.join(top_cat)

    base = pd.DataFrame({"account_id": all_account_ids}).set_index("account_id")
    result = base.join(per_acct).fillna(0)

    return result[[
        "merch_unique_count", "merch_concentration", "merch_max_accts_per_merchant",
        "merch_top_category", "merch_avg_age_days",
    ]]


# ---------------------------------------------------------------------------
# 3f. Gene Group — Graph
# ---------------------------------------------------------------------------

def compute_graph_genes(account_features_df, community_stats_df, all_account_ids):
    """Graph-structural and community features from community.py outputs."""
    feat = account_features_df[[
        "account_id", "degree", "weighted_degree",
        "connected_device_count", "connected_ip_count", "connected_merchant_count",
        "shared_device_count", "shared_ip_count",
        "proj_degree", "proj_weighted_degree", "community_id_louvain",
    ]].copy().rename(columns={
        "degree":                   "graph_degree",
        "weighted_degree":          "graph_weighted_degree",
        "connected_device_count":   "graph_connected_devices",
        "connected_ip_count":       "graph_connected_ips",
        "connected_merchant_count": "graph_connected_merchants",
        "shared_device_count":      "graph_shared_devices",
        "shared_ip_count":          "graph_shared_ips",
        "proj_degree":              "graph_proj_degree",
        "proj_weighted_degree":     "graph_proj_weighted_degree",
    })

    # Join community-level stats by Louvain community assignment
    comm_lookup = community_stats_df[[
        "community_id", "size", "density",
        "fraud_ratio", "shared_device_ratio", "shared_ip_ratio",
    ]].rename(columns={
        "community_id":         "community_id_louvain",
        "size":                 "graph_community_size",
        "density":              "graph_community_density",
        "fraud_ratio":          "graph_community_fraud_ratio",
        "shared_device_ratio":  "graph_community_shared_device_ratio",
        "shared_ip_ratio":      "graph_community_shared_ip_ratio",
    })

    feat = feat.merge(comm_lookup, on="community_id_louvain", how="left")
    feat = feat.drop(columns=["community_id_louvain"]).set_index("account_id")

    base = pd.DataFrame({"account_id": all_account_ids}).set_index("account_id")
    result = base.join(feat).fillna(0)

    return result[[
        "graph_degree", "graph_weighted_degree",
        "graph_connected_devices", "graph_connected_ips", "graph_connected_merchants",
        "graph_shared_devices", "graph_shared_ips",
        "graph_proj_degree", "graph_proj_weighted_degree",
        "graph_community_size", "graph_community_density",
        "graph_community_fraud_ratio",
        "graph_community_shared_device_ratio", "graph_community_shared_ip_ratio",
    ]]


# ---------------------------------------------------------------------------
# 4. Assembly
# ---------------------------------------------------------------------------

def assemble_genome(
    phase_label, phase_df, accounts_df, merchants_df,
    account_features_df, community_stats_df,
    all_account_ids, phase_start, phase_end,
):
    """Merge all six gene groups into a single per-account Genome DataFrame."""
    phase_days = max(int((phase_end - phase_start).days), 1)

    # Per-phase fraud metadata
    fraud_meta = (
        phase_df[phase_df["is_fraud"]]
        .groupby("account_id")
        .agg(is_fraud_account=("is_fraud", "any"),
             fraud_cluster_id=("fraud_cluster_id", "first"))
    )

    print(f"  [{phase_label}] Computing Account Genes...")
    g_acct = compute_account_genes(phase_df, accounts_df, all_account_ids, phase_start, phase_days)

    print(f"  [{phase_label}] Computing Device Genes...")
    g_dev = compute_device_genes(phase_df.copy(), all_account_ids)

    print(f"  [{phase_label}] Computing Network Genes...")
    g_net = compute_network_genes(phase_df.copy(), all_account_ids)

    print(f"  [{phase_label}] Computing Transaction Genes...")
    g_tx = compute_transaction_genes(phase_df.copy(), all_account_ids, phase_days)

    print(f"  [{phase_label}] Computing Merchant Genes...")
    g_merch = compute_merchant_genes(phase_df.copy(), merchants_df, all_account_ids, phase_start)

    print(f"  [{phase_label}] Computing Graph Genes...")
    g_graph = compute_graph_genes(account_features_df, community_stats_df, all_account_ids)

    # Merge all gene groups
    genome = (
        pd.DataFrame({"account_id": all_account_ids}).set_index("account_id")
        .join([g_acct, g_dev, g_net, g_tx, g_merch, g_graph])
    )

    # Attach metadata
    genome["phase"] = phase_label
    genome = genome.join(fraud_meta[["is_fraud_account", "fraud_cluster_id"]])
    genome["is_fraud_account"] = genome["is_fraud_account"].fillna(False)
    genome["fraud_cluster_id"] = genome["fraud_cluster_id"].fillna("none")

    return genome.reset_index()


# ---------------------------------------------------------------------------
# 5. Normalization
# ---------------------------------------------------------------------------

def normalize_genome(genome_full_df):
    """Global min-max normalize all continuous gene features across all phases."""
    continuous_cols = [
        c for c in genome_full_df.columns
        if any(c.startswith(p) for p in GENE_PREFIXES) and c not in NON_SCALED_COLS
    ]

    genome_norm = genome_full_df.copy()
    for col in continuous_cols:
        col_min = genome_full_df[col].min()
        col_max = genome_full_df[col].max()
        if col_max > col_min:
            genome_norm[col] = (genome_full_df[col] - col_min) / (col_max - col_min)
        else:
            genome_norm[col] = 0.0

    return genome_norm


# ---------------------------------------------------------------------------
# 6. Gene Profile Summary
# ---------------------------------------------------------------------------

def print_gene_profile_summary(genome_norm_df):
    """Print a fraud-vs-normal gene profile comparison across T2 and T3."""
    gene_groups = {
        "ACCOUNT":     [c for c in genome_norm_df.columns if c.startswith("acct_")],
        "DEVICE":      [c for c in genome_norm_df.columns if c.startswith("dev_")],
        "NETWORK":     [c for c in genome_norm_df.columns if c.startswith("net_")],
        "TRANSACTION": [c for c in genome_norm_df.columns if c.startswith("tx_")],
        "MERCHANT":    [c for c in genome_norm_df.columns if c.startswith("merch_")],
        "GRAPH":       [c for c in genome_norm_df.columns if c.startswith("graph_")],
    }

    t2f = genome_norm_df[(genome_norm_df["phase"] == "T2") & genome_norm_df["is_fraud_account"]]
    t2n = genome_norm_df[(genome_norm_df["phase"] == "T2") & ~genome_norm_df["is_fraud_account"]]
    t3f = genome_norm_df[(genome_norm_df["phase"] == "T3") & genome_norm_df["is_fraud_account"]]
    t3n = genome_norm_df[(genome_norm_df["phase"] == "T3") & ~genome_norm_df["is_fraud_account"]]

    bar = "=" * 90
    print(f"\n{bar}")
    print("  Behavioral Genome — Gene Profile Summary  (normalized 0→1, mean per group)")
    print(bar)
    print(f"  {'Group':<13} {'Feature':<43} {'T2 Fraud':>9} {'T2 Normal':>10} {'T3 Fraud':>9} {'T3 Normal':>10}")
    print(f"  {'-'*88}")

    for group_name, cols in gene_groups.items():
        first = True
        for col in cols:
            v_t2f = t2f[col].mean() if not t2f.empty else 0.0
            v_t2n = t2n[col].mean() if not t2n.empty else 0.0
            v_t3f = t3f[col].mean() if not t3f.empty else 0.0
            v_t3n = t3n[col].mean() if not t3n.empty else 0.0

            fraud_signal  = abs(v_t2f - v_t2n) > 0.05 or abs(v_t3f - v_t3n) > 0.05
            mutation      = abs(v_t2f - v_t3f) > 0.05

            marker = ""
            if fraud_signal and mutation:
                marker = " ↓"
            elif fraud_signal:
                marker = " ✦"

            g_label = group_name if first else ""
            first = False
            print(
                f"  {g_label:<13} {col:<43} {v_t2f:>9.3f} {v_t2n:>10.3f} "
                f"{v_t3f:>9.3f}{marker:<3} {v_t3n:>10.3f}"
            )
        print()

    print(f"  Legend:  ✦ = strong fraud signal (stable T2→T3)   ↓ = signal mutates T2→T3")
    print(f"{bar}\n")


# ---------------------------------------------------------------------------
# 7. Save
# ---------------------------------------------------------------------------

def save_genome(genome_full_df, genome_norm_df, data_dir):
    """Save raw and normalized genome DataFrames per phase and combined."""
    for phase in ("T1", "T2", "T3"):
        lbl = phase.lower()
        genome_full_df[genome_full_df["phase"] == phase].to_csv(
            os.path.join(data_dir, f"genome_{lbl}_raw.csv"), index=False
        )
        genome_norm_df[genome_norm_df["phase"] == phase].to_csv(
            os.path.join(data_dir, f"genome_{lbl}.csv"), index=False
        )

    genome_full_df.to_csv(os.path.join(data_dir, "genome_full_raw.csv"), index=False)
    genome_norm_df.to_csv(os.path.join(data_dir, "genome_full.csv"), index=False)

    n_rows, n_cols = genome_norm_df.shape
    print(f"  Saved genome_full.csv ({n_rows} rows × {n_cols} columns)")
    print(f"  Saved genome_t1.csv, genome_t2.csv, genome_t3.csv  [normalized]")
    print(f"  Saved genome_*_raw.csv  [pre-normalization]")


# ---------------------------------------------------------------------------
# 8. Main Pipeline
# ---------------------------------------------------------------------------

def main():
    merged_df, accounts_df, merchants_df, graph_features, community_stats, data_dir = (
        load_all_inputs()
    )

    phase_windows = determine_phase_windows(merged_df)
    all_account_ids = accounts_df["account_id"].tolist()

    phase_slices = {
        "T1": merged_df[merged_df["timestamp_dt"] < phase_windows["T1"][1]],
        "T2": merged_df[
            (merged_df["timestamp_dt"] >= phase_windows["T2"][0])
            & (merged_df["timestamp_dt"] < phase_windows["T2"][1])
        ],
        "T3": merged_df[merged_df["timestamp_dt"] >= phase_windows["T3"][0]],
    }

    genome_phases = []
    for phase_label in ("T1", "T2", "T3"):
        phase_df = phase_slices[phase_label]
        phase_start, phase_end = phase_windows[phase_label]

        print(f"\n{'=' * 58}")
        print(f"  Assembling Genome: Phase {phase_label}  ({len(phase_df)} transactions)")
        print(f"{'=' * 58}")

        genome_phase = assemble_genome(
            phase_label=phase_label,
            phase_df=phase_df,
            accounts_df=accounts_df,
            merchants_df=merchants_df,
            account_features_df=graph_features[phase_label],
            community_stats_df=community_stats[phase_label],
            all_account_ids=all_account_ids,
            phase_start=phase_start,
            phase_end=phase_end,
        )
        genome_phases.append(genome_phase)
        print(
            f"  → {len(genome_phase)} accounts  ×  "
            f"{len(genome_phase.columns)} columns  "
            f"({genome_phase['is_fraud_account'].sum()} fraud accounts)"
        )

    # Stack all phases
    genome_full = pd.concat(genome_phases, ignore_index=True)
    print(f"\nFull genome shape before mutation features: {genome_full.shape}")

    # Compute mutation-aware higher level features
    print("Computing mutation-aware behavioral & relational features...")
    genome_full = compute_mutation_aware_features(genome_full, data_dir)
    print(f"Full genome shape with mutation features: {genome_full.shape}")

    # Global normalization
    print("Normalizing features globally across all phases...")
    genome_norm = normalize_genome(genome_full)

    # Gene profile summary
    print_gene_profile_summary(genome_norm)

    # Save all outputs
    print("Saving genome outputs...")
    save_genome(genome_full, genome_norm, data_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
