import json
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx


def load_data(data_dir=None):
    """Load transactions, fraud labels, and entity metadata CSV files."""
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")

    print(f"Loading data from '{data_dir}'...")
    transactions_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    labels_df = pd.read_csv(os.path.join(data_dir, "fraud_labels.csv"))
    accounts_df = pd.read_csv(os.path.join(data_dir, "accounts.csv"))
    merchants_df = pd.read_csv(os.path.join(data_dir, "merchants.csv"))
    devices_df = pd.read_csv(os.path.join(data_dir, "devices.csv"))
    ips_df = pd.read_csv(os.path.join(data_dir, "ips.csv"))

    # Merge transactions with fraud labels
    merged_df = pd.merge(transactions_df, labels_df, on="transaction_id")
    merged_df["timestamp_dt"] = pd.to_datetime(merged_df["timestamp"])

    entity_dfs = {
        "accounts": accounts_df,
        "merchants": merchants_df,
        "devices": devices_df,
        "ips": ips_df,
    }

    return merged_df, entity_dfs, data_dir


def build_behavioral_graph(merged_df, entity_dfs):
    """Build a NetworkX Graph representing behavioral relationships between entities.

    Nodes:
      - Account (account:<id>)
      - Device (device:<id>)
      - IP (ip:<id>)
      - Merchant (merchant:<id>)

    Edges:
      - Account ── Device
      - Account ── IP
      - Account ── Merchant
    """
    G = nx.Graph()

    # 1. Add entity nodes with metadata
    for _, row in entity_dfs["accounts"].iterrows():
        node_id = f"account:{row['account_id']}"
        G.add_node(
            node_id,
            node_type="account",
            raw_id=row["account_id"],
            created_at=row.get("created_at"),
            country=row.get("country"),
            kyc_status=row.get("kyc_status"),
        )

    for _, row in entity_dfs["devices"].iterrows():
        node_id = f"device:{row['device_id']}"
        G.add_node(
            node_id,
            node_type="device",
            raw_id=row["device_id"],
            device_type=row.get("device_type"),
        )

    for _, row in entity_dfs["ips"].iterrows():
        node_id = f"ip:{row['ip_id']}"
        G.add_node(
            node_id,
            node_type="ip",
            raw_id=row["ip_id"],
            ip_address=row.get("ip_address"),
        )

    for _, row in entity_dfs["merchants"].iterrows():
        node_id = f"merchant:{row['merchant_id']}"
        G.add_node(
            node_id,
            node_type="merchant",
            raw_id=row["merchant_id"],
            created_at=row.get("created_at"),
            category=row.get("category"),
        )

    # 2. Vectorized relationship triples generation
    acct_nodes = "account:" + merged_df["account_id"].astype(str)
    dev_nodes = "device:" + merged_df["device_id"].astype(str)
    ip_nodes = "ip:" + merged_df["ip_id"].astype(str)
    merch_nodes = "merchant:" + merged_df["merchant_id"].astype(str)

    timestamps = merged_df["timestamp_dt"].values
    amounts = merged_df["amount"].astype(float).values
    is_frauds = merged_df["is_fraud"].astype(bool).values

    sources = np.concatenate([acct_nodes, acct_nodes, acct_nodes])
    targets = np.concatenate([dev_nodes, ip_nodes, merch_nodes])
    edge_types = np.concatenate(
        [
            np.full(len(merged_df), "account_device"),
            np.full(len(merged_df), "account_ip"),
            np.full(len(merged_df), "account_merchant"),
        ]
    )
    edge_ts = np.concatenate([timestamps, timestamps, timestamps])
    edge_amt = np.concatenate([amounts, amounts, amounts])
    edge_fraud = np.concatenate([is_frauds, is_frauds, is_frauds])

    edges_df = pd.DataFrame(
        {
            "source": sources,
            "target": targets,
            "edge_type": edge_types,
            "timestamp": edge_ts,
            "amount": edge_amt,
            "is_fraud": edge_fraud,
        }
    )

    aggregated = edges_df.groupby(["source", "target", "edge_type"]).agg(
        transaction_count=("timestamp", "count"),
        first_seen=("timestamp", "min"),
        last_seen=("timestamp", "max"),
        total_amount=("amount", "sum"),
        is_fraud=("is_fraud", "any"),
        fraud_transaction_count=("is_fraud", "sum"),
        timestamps=("timestamp", list),
    ).reset_index()

    for row in aggregated.itertuples(index=False):
        src, tgt = row.source, row.target
        if not G.has_node(src):
            G.add_node(src, node_type=src.split(":")[0], raw_id=src.split(":", 1)[1])
        if not G.has_node(tgt):
            G.add_node(tgt, node_type=tgt.split(":")[0], raw_id=tgt.split(":", 1)[1])

        ts_min = pd.to_datetime(row.first_seen).isoformat()
        ts_max = pd.to_datetime(row.last_seen).isoformat()

        G.add_edge(
            src,
            tgt,
            edge_type=row.edge_type,
            weight=int(row.transaction_count),
            transaction_count=int(row.transaction_count),
            first_seen=ts_min,
            last_seen=ts_max,
            total_amount=round(float(row.total_amount), 2),
            timestamps=row.timestamps,
            is_fraud=bool(row.is_fraud),
            fraud_transaction_count=int(row.fraud_transaction_count),
        )

    return G


def calculate_temporal_decay(G, reference_time=None, decay_lambda=0.1):
    """Calculate exponentially decayed edge weights: w_ij(t) = sum_k exp(-lambda * (t - t_k)).

    where (t - t_k) is measured in days relative to reference_time.
    """
    if reference_time is None:
        max_ts_list = []
        for u, v, d in G.edges(data=True):
            if "timestamps" in d and len(d["timestamps"]) > 0:
                max_ts_list.append(np.max(d["timestamps"]))
        if max_ts_list:
            reference_time = pd.to_datetime(np.max(max_ts_list))
        else:
            reference_time = datetime.now()
    elif isinstance(reference_time, str):
        reference_time = pd.to_datetime(reference_time)

    ref_dt64 = np.datetime64(reference_time)

    for u, v, d in G.edges(data=True):
        if "timestamps" in d and len(d["timestamps"]) > 0:
            ts_array = np.asarray(d["timestamps"], dtype="datetime64[ns]")
            delta_days = (ref_dt64 - ts_array).astype("timedelta64[ns]").astype(float) / (
                86400.0 * 1e9
            )
            valid_deltas = delta_days[delta_days >= 0]
            decayed_weight = float(np.sum(np.exp(-decay_lambda * valid_deltas)))
            d["decayed_weight"] = round(decayed_weight, 4)
        else:
            d["decayed_weight"] = float(d.get("weight", 1.0))

    return G


def compute_graph_statistics(G):
    """Compute and print basic graph statistics."""
    stats = {}

    nodes_by_type = {}
    for n, d in G.nodes(data=True):
        ntype = d.get("node_type", "unknown")
        nodes_by_type[ntype] = nodes_by_type.get(ntype, 0) + 1

    edges_by_type = {}
    for u, v, d in G.edges(data=True):
        etype = d.get("edge_type", "unknown")
        edges_by_type[etype] = edges_by_type.get(etype, 0) + 1

    stats["num_nodes"] = G.number_of_nodes()
    stats["num_edges"] = G.number_of_edges()
    stats["nodes_by_type"] = nodes_by_type
    stats["edges_by_type"] = edges_by_type
    stats["density"] = round(nx.density(G), 6)
    stats["num_connected_components"] = nx.number_connected_components(G)

    print("\n================ Graph Statistics ================")
    print(f"Total Nodes: {stats['num_nodes']}")
    for ntype, count in nodes_by_type.items():
        print(f"  - {ntype.capitalize()} Nodes: {count}")
    print(f"Total Edges: {stats['num_edges']}")
    for etype, count in edges_by_type.items():
        print(f"  - {etype} Edges: {count}")
    print(f"Graph Density: {stats['density']}")
    print(f"Connected Components: {stats['num_connected_components']}")
    print("==================================================\n")

    return stats


def extract_phase_subgraphs(merged_df, entity_dfs):
    """Split transactions into T1, T2, T3 phases and build a network for each."""
    min_time = merged_df["timestamp_dt"].min()
    max_time = merged_df["timestamp_dt"].max()
    total_duration = max_time - min_time
    phase_duration = total_duration / 3

    t1_end = min_time + phase_duration
    t2_end = min_time + 2 * phase_duration

    t1_df = merged_df[merged_df["timestamp_dt"] < t1_end]
    t2_df = merged_df[(merged_df["timestamp_dt"] >= t1_end) & (merged_df["timestamp_dt"] < t2_end)]
    t3_df = merged_df[merged_df["timestamp_dt"] >= t2_end]

    print(f"Phase T1 transactions: {len(t1_df)}")
    print(f"Phase T2 transactions: {len(t2_df)}")
    print(f"Phase T3 transactions: {len(t3_df)}")

    G_t1 = build_behavioral_graph(t1_df, entity_dfs)
    G_t2 = build_behavioral_graph(t2_df, entity_dfs)
    G_t3 = build_behavioral_graph(t3_df, entity_dfs)

    calculate_temporal_decay(G_t1, reference_time=t1_end)
    calculate_temporal_decay(G_t2, reference_time=t2_end)
    calculate_temporal_decay(G_t3, reference_time=max_time)

    return G_t1, G_t2, G_t3


def verify_fraud_ring_topology(G_t2, G_t3, merged_df):
    """Verify that the synthetic fraud ring produces the exact intended topology in T2 and T3.

    T2 Fraud Cluster Expected:
      - 10 accounts
      - 3 shared devices
      - 2 shared IPs
      - 1 shared merchant

    T3 Fraud Cluster Expected:
      - 10 accounts
      - 10 mutated devices (1 per account)
      - 2 shared IPs (retained)
      - 1 shared merchant (retained)
    """
    print("\n--- Verifying Fraud Ring Graph Topology (T2 vs T3) ---")

    t2_fraud_tx = merged_df[merged_df["scenario"] == "t2_coordinated_fraud"]

    fraud_account_ids = sorted(t2_fraud_tx["account_id"].unique())
    fraud_account_nodes = [f"account:{aid}" for aid in fraud_account_ids]

    print(f"Target Fraud Ring Accounts ({len(fraud_account_nodes)} accounts):")

    # Analyze T2 Fraud Cluster Topology (filtering for edges carrying fraud activity)
    t2_fraud_devices, t2_fraud_ips, t2_fraud_merchants = set(), set(), set()
    for acct_node in fraud_account_nodes:
        if G_t2.has_node(acct_node):
            for neighbor in G_t2.neighbors(acct_node):
                edge_data = G_t2.get_edge_data(acct_node, neighbor)
                if edge_data.get("is_fraud"):
                    ntype = G_t2.nodes[neighbor].get("node_type")
                    if ntype == "device":
                        t2_fraud_devices.add(neighbor)
                    elif ntype == "ip":
                        t2_fraud_ips.add(neighbor)
                    elif ntype == "merchant":
                        t2_fraud_merchants.add(neighbor)

    print("\n[Phase T2 Coordinated Fraud Cluster Subgraph Topology]")
    print(f"  - Fraud Accounts: {len(fraud_account_nodes)}")
    print(f"  - Connected Shared Devices: {len(t2_fraud_devices)} (Expected: 3)")
    print(f"  - Connected Shared IPs:     {len(t2_fraud_ips)} (Expected: 2)")
    print(f"  - Connected Shared Merchant:{len(t2_fraud_merchants)} (Expected: 1)")

    assert len(fraud_account_nodes) == 10, f"Expected 10 fraud accounts, got {len(fraud_account_nodes)}"
    assert len(t2_fraud_devices) == 3, f"T2 fraud cluster failed: expected 3 shared devices, got {len(t2_fraud_devices)}"
    assert len(t2_fraud_ips) == 2, f"T2 fraud cluster failed: expected 2 shared IPs, got {len(t2_fraud_ips)}"
    assert len(t2_fraud_merchants) == 1, f"T2 fraud cluster failed: expected 1 merchant, got {len(t2_fraud_merchants)}"
    print("✓ T2 Coordinated Fraud Cluster Topology Proof PASSED!")

    # Analyze T3 Fraud Cluster Topology (filtering for edges carrying fraud activity)
    t3_fraud_devices, t3_fraud_ips, t3_fraud_merchants = set(), set(), set()
    for acct_node in fraud_account_nodes:
        if G_t3.has_node(acct_node):
            for neighbor in G_t3.neighbors(acct_node):
                edge_data = G_t3.get_edge_data(acct_node, neighbor)
                if edge_data.get("is_fraud"):
                    ntype = G_t3.nodes[neighbor].get("node_type")
                    if ntype == "device":
                        t3_fraud_devices.add(neighbor)
                    elif ntype == "ip":
                        t3_fraud_ips.add(neighbor)
                    elif ntype == "merchant":
                        t3_fraud_merchants.add(neighbor)

    print("\n[Phase T3 Mutated Fraud Cluster Subgraph Topology]")
    print(f"  - Fraud Accounts: {len(fraud_account_nodes)}")
    print(f"  - Connected Mutated Devices: {len(t3_fraud_devices)} (Expected: 10)")
    print(f"  - Connected Retained IPs:    {len(t3_fraud_ips)} (Expected: 2)")
    print(f"  - Connected Retained Merchant:{len(t3_fraud_merchants)} (Expected: 1)")

    assert len(t3_fraud_devices) == 10, f"T3 fraud cluster failed: expected 10 distinct devices, got {len(t3_fraud_devices)}"
    assert len(t3_fraud_ips) == 2, f"T3 fraud cluster failed: expected 2 shared IPs, got {len(t3_fraud_ips)}"
    assert len(t3_fraud_merchants) == 1, f"T3 fraud cluster failed: expected 1 merchant, got {len(t3_fraud_merchants)}"
    assert t2_fraud_merchants == t3_fraud_merchants, "T2 and T3 fraud merchants do not match!"
    assert t2_fraud_ips == t3_fraud_ips, "T2 and T3 fraud IPs do not match!"
    print("✓ T3 Mutated Fraud Cluster Topology Proof PASSED!")

    print("\n--- All Topological Verification Proofs PASSED Successfully! ---\n")


def prepare_graph_for_saving(G):
    """Convert numpy datetime objects on edge attributes to ISO strings for clean pickling/JSON."""
    G_copy = G.copy()
    for u, v, d in G_copy.edges(data=True):
        if "timestamps" in d and isinstance(d["timestamps"], (list, np.ndarray)):
            d["timestamps"] = [pd.to_datetime(ts).isoformat() for ts in d["timestamps"]]
    return G_copy


def save_graphs(G_full, G_t1, G_t2, G_t3, stats, data_dir):
    """Save graph objects as pickle files and summary stats as JSON."""
    with open(os.path.join(data_dir, "graph_full.pkl"), "wb") as f:
        pickle.dump(prepare_graph_for_saving(G_full), f)

    with open(os.path.join(data_dir, "graph_t1.pkl"), "wb") as f:
        pickle.dump(prepare_graph_for_saving(G_t1), f)

    with open(os.path.join(data_dir, "graph_t2.pkl"), "wb") as f:
        pickle.dump(prepare_graph_for_saving(G_t2), f)

    with open(os.path.join(data_dir, "graph_t3.pkl"), "wb") as f:
        pickle.dump(prepare_graph_for_saving(G_t3), f)

    with open(os.path.join(data_dir, "graph_summary.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved graph objects and summary stats to '{data_dir}'.")


def main():
    merged_df, entity_dfs, data_dir = load_data()

    print("\nBuilding full multi-entity behavioral graph...")
    G_full = build_behavioral_graph(merged_df, entity_dfs)
    calculate_temporal_decay(G_full)

    stats = compute_graph_statistics(G_full)

    print("Extracting phase-specific subgraphs (T1, T2, T3)...")
    G_t1, G_t2, G_t3 = extract_phase_subgraphs(merged_df, entity_dfs)

    verify_fraud_ring_topology(G_t2, G_t3, merged_df)

    save_graphs(G_full, G_t1, G_t2, G_t3, stats, data_dir)


if __name__ == "__main__":
    main()
