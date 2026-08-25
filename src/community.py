import json
import os
import pickle
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms import community as nx_community


# ---------------------------------------------------------------------------
# 1. Data Loading
# ---------------------------------------------------------------------------

def load_phase_graphs(data_dir=None):
    """Load pre-built phase graphs from build_graph.py and raw transaction data."""
    if data_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(project_root, "data")

    print(f"Loading phase graphs from '{data_dir}'...")
    with open(os.path.join(data_dir, "graph_t1.pkl"), "rb") as f:
        G_t1 = pickle.load(f)
    with open(os.path.join(data_dir, "graph_t2.pkl"), "rb") as f:
        G_t2 = pickle.load(f)
    with open(os.path.join(data_dir, "graph_t3.pkl"), "rb") as f:
        G_t3 = pickle.load(f)

    transactions_df = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    labels_df = pd.read_csv(os.path.join(data_dir, "fraud_labels.csv"))
    merged_df = pd.merge(transactions_df, labels_df, on="transaction_id")
    merged_df["timestamp_dt"] = pd.to_datetime(merged_df["timestamp"])

    return G_t1, G_t2, G_t3, merged_df, data_dir


# ---------------------------------------------------------------------------
# 2. Graph Projection: Heterogeneous → Account-Account
# ---------------------------------------------------------------------------

def project_to_account_graph(G):
    """Project the heterogeneous behavioral graph onto a pure account-account graph.

    Two account nodes are connected if they share at least one Device, IP, or Merchant
    entity. Edge weight encodes behavioral similarity strength:

        weight = shared_device_count + shared_ip_count + shared_merchant_count

    This projection prevents naive entity-type communities from dominating Louvain
    and focuses community structure on coordinated account behavior.

    Args:
        G: Heterogeneous NetworkX graph from build_graph.py.

    Returns:
        G_proj: Account-only NetworkX Graph with weighted edges.
    """
    # Build entity_node → {account_node, ...} mapping
    entity_to_accounts = defaultdict(set)
    for u, v, d in G.edges(data=True):
        u_type = G.nodes[u].get("node_type")
        v_type = G.nodes[v].get("node_type")

        if u_type == "account" and v_type != "account":
            entity_to_accounts[v].add(u)
        elif v_type == "account" and u_type != "account":
            entity_to_accounts[u].add(v)
        # account-account edges do not exist in the heterogeneous graph by construction

    # Build account-account projection graph
    G_proj = nx.Graph()

    # Add all account nodes (even isolated ones)
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "account":
            G_proj.add_node(
                node,
                node_type="account",
                raw_id=data.get("raw_id"),
                kyc_status=data.get("kyc_status"),
                country=data.get("country"),
            )

    # For each entity with ≥2 accounts: add/accumulate account-account edges
    for entity_node, acct_set in entity_to_accounts.items():
        if len(acct_set) < 2:
            continue

        entity_type = G.nodes[entity_node].get("node_type")  # device / ip / merchant
        acct_list = list(acct_set)

        for a1, a2 in combinations(acct_list, 2):
            if G_proj.has_edge(a1, a2):
                d = G_proj[a1][a2]
                d["weight"] += 1
                d[f"shared_{entity_type}_count"] = d.get(f"shared_{entity_type}_count", 0) + 1
            else:
                G_proj.add_edge(
                    a1, a2,
                    weight=1,
                    shared_device_count=1 if entity_type == "device" else 0,
                    shared_ip_count=1 if entity_type == "ip" else 0,
                    shared_merchant_count=1 if entity_type == "merchant" else 0,
                )

    print(
        f"  Account projection: {G_proj.number_of_nodes()} account nodes, "
        f"{G_proj.number_of_edges()} account-account edges"
    )
    return G_proj


# ---------------------------------------------------------------------------
# 3. Community Detection
# ---------------------------------------------------------------------------

def detect_communities(G_proj, methods=("louvain", "lpa"), seed=42):
    """Run community detection algorithms on the account-account projection graph.

    Args:
        G_proj: Account-account NetworkX Graph (output of project_to_account_graph).
        methods: Tuple of method names. Supported: 'louvain', 'lpa'.
        seed: Random seed for reproducibility.

    Returns:
        dict: {method_name: {account_node: community_id (int)}}
    """
    results = {}

    for method in methods:
        print(f"  Running {method.upper()} community detection...")

        if method == "louvain":
            community_sets = nx_community.louvain_communities(
                G_proj, weight="weight", seed=seed
            )
        elif method == "lpa":
            community_sets = nx_community.fast_label_propagation_communities(
                G_proj, weight="weight", seed=seed
            )
        else:
            raise ValueError(f"Unsupported community detection method: '{method}'")

        # Convert list-of-sets → {node: community_id}
        node_to_community = {}
        for cid, node_set in enumerate(community_sets):
            for node in node_set:
                node_to_community[node] = cid

        # Assign -1 to any isolated nodes not assigned a community
        for node in G_proj.nodes():
            if node not in node_to_community:
                node_to_community[node] = -1

        n_communities = len(set(node_to_community.values()))
        print(f"    → {n_communities} communities detected by {method.upper()}")
        results[method] = node_to_community

    return results


# ---------------------------------------------------------------------------
# 4. Community Statistics
# ---------------------------------------------------------------------------

def compute_community_stats(
    community_id_map, G_hetero, G_proj, merged_df, phase_label
):
    """Compute per-community aggregate statistics.

    Args:
        community_id_map: {account_node: community_id} from detect_communities.
        G_hetero: Full heterogeneous behavioral graph for this phase.
        G_proj: Account-account projection for this phase.
        merged_df: Transactions + labels DataFrame for this phase.
        phase_label: String identifier e.g. 'T2'.

    Returns:
        pd.DataFrame: One row per community with aggregate statistics.
    """
    # Build community → [account_nodes]
    community_to_accounts = defaultdict(list)
    for node, cid in community_id_map.items():
        community_to_accounts[cid].append(node)

    # Build account → is_fraud_account flag
    fraud_accounts = set(
        "account:" + merged_df[merged_df["is_fraud"]]["account_id"].unique()
    )

    rows = []
    for cid, acct_nodes in community_to_accounts.items():
        size = len(acct_nodes)
        acct_node_set = set(acct_nodes)

        # Fraud ratio
        fraud_in_community = sum(1 for a in acct_nodes if a in fraud_accounts)
        fraud_ratio = fraud_in_community / size if size > 0 else 0.0

        # Entity sets for this community
        devices_in_comm = set()
        ips_in_comm = set()
        merchants_in_comm = set()

        # Device/IP/Merchant counts and sharing
        entity_account_count = defaultdict(set)  # entity_node → set of accounts using it

        for acct_node in acct_nodes:
            if not G_hetero.has_node(acct_node):
                continue
            for neighbor in G_hetero.neighbors(acct_node):
                ntype = G_hetero.nodes[neighbor].get("node_type")
                if ntype == "device":
                    devices_in_comm.add(neighbor)
                    entity_account_count[neighbor].add(acct_node)
                elif ntype == "ip":
                    ips_in_comm.add(neighbor)
                    entity_account_count[neighbor].add(acct_node)
                elif ntype == "merchant":
                    merchants_in_comm.add(neighbor)
                    entity_account_count[neighbor].add(acct_node)

        unique_devices = len(devices_in_comm)
        unique_ips = len(ips_in_comm)
        unique_merchants = len(merchants_in_comm)

        # Shared entity counts (shared = used by ≥2 accounts in community)
        shared_devices = sum(1 for d, accts in entity_account_count.items()
                             if G_hetero.nodes[d]["node_type"] == "device" and len(accts) >= 2)
        shared_ips = sum(1 for d, accts in entity_account_count.items()
                         if G_hetero.nodes[d]["node_type"] == "ip" and len(accts) >= 2)

        shared_device_ratio = shared_devices / unique_devices if unique_devices > 0 else 0.0
        shared_ip_ratio = shared_ips / unique_ips if unique_ips > 0 else 0.0

        # Projection subgraph density
        subgraph = G_proj.subgraph(acct_node_set)
        density = nx.density(subgraph) if size >= 2 else 0.0

        # Average projection degree within community
        avg_proj_degree = (
            sum(G_proj.degree(n) for n in acct_nodes if G_proj.has_node(n)) / size
            if size > 0 else 0.0
        )

        rows.append({
            "phase": phase_label,
            "community_id": cid,
            "size": size,
            "fraud_account_count": fraud_in_community,
            "fraud_ratio": round(fraud_ratio, 4),
            "density": round(density, 4),
            "unique_devices": unique_devices,
            "unique_ips": unique_ips,
            "unique_merchants": unique_merchants,
            "shared_device_count": shared_devices,
            "shared_ip_count": shared_ips,
            "shared_device_ratio": round(shared_device_ratio, 4),
            "shared_ip_ratio": round(shared_ip_ratio, 4),
            "avg_proj_degree": round(avg_proj_degree, 2),
        })

    df = pd.DataFrame(rows).sort_values("size", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 5. Account-Level Graph Features
# ---------------------------------------------------------------------------

def compute_account_features(G_hetero, G_proj, community_maps, merged_df, phase_label):
    """Compute per-account graph-derived behavioral features.

    Args:
        G_hetero: Full heterogeneous behavioral graph for this phase.
        G_proj: Account-account projection for this phase.
        community_maps: {method: {account_node: community_id}}.
        merged_df: Transactions + labels DataFrame for this phase.
        phase_label: String identifier e.g. 'T2'.

    Returns:
        pd.DataFrame: One row per account with graph-derived features.
    """
    fraud_accounts = set(
        "account:" + merged_df[merged_df["is_fraud"]]["account_id"].unique()
    )

    # Build entity → accounts set to calculate sharing
    entity_to_accounts = defaultdict(set)
    for u, v in G_hetero.edges():
        u_type = G_hetero.nodes[u].get("node_type")
        v_type = G_hetero.nodes[v].get("node_type")
        if u_type == "account" and v_type != "account":
            entity_to_accounts[v].add(u)
        elif v_type == "account" and u_type != "account":
            entity_to_accounts[u].add(v)

    # Entity nodes shared by ≥2 accounts
    shared_entities = {e for e, accts in entity_to_accounts.items() if len(accts) >= 2}

    rows = []
    for node, data in G_hetero.nodes(data=True):
        if data.get("node_type") != "account":
            continue

        # Neighbor sets by type
        devices, ips, merchants = [], [], []
        weighted_deg = 0.0
        for neighbor in G_hetero.neighbors(node):
            ntype = G_hetero.nodes[neighbor].get("node_type")
            edge_w = G_hetero[node][neighbor].get("weight", 1)
            weighted_deg += edge_w
            if ntype == "device":
                devices.append(neighbor)
            elif ntype == "ip":
                ips.append(neighbor)
            elif ntype == "merchant":
                merchants.append(neighbor)

        shared_dev_count = sum(1 for d in devices if d in shared_entities)
        shared_ip_count = sum(1 for ip in ips if ip in shared_entities)

        # Projection degree = number of accounts sharing ≥1 entity with this account
        proj_degree = G_proj.degree(node) if G_proj.has_node(node) else 0
        proj_weighted_degree = (
            sum(d.get("weight", 1) for _, d in G_proj[node].items())
            if G_proj.has_node(node) else 0
        )

        row = {
            "phase": phase_label,
            "account_node": node,
            "account_id": data.get("raw_id"),
            "is_fraud_account": node in fraud_accounts,
            "degree": G_hetero.degree(node),
            "weighted_degree": round(weighted_deg, 2),
            "connected_device_count": len(devices),
            "connected_ip_count": len(ips),
            "connected_merchant_count": len(merchants),
            "shared_device_count": shared_dev_count,
            "shared_ip_count": shared_ip_count,
            "proj_degree": proj_degree,
            "proj_weighted_degree": proj_weighted_degree,
        }

        # Attach community assignments from each detection method
        for method, id_map in community_maps.items():
            row[f"community_id_{method}"] = id_map.get(node, -1)

        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Topological Proof: Ring Co-Clustering Verification
# ---------------------------------------------------------------------------

def verify_ring_community(community_id_map, merged_df, phase_label, method_name):
    """Assert that the 10 fraud-ring accounts cluster together in the community graph.

    Args:
        community_id_map: {account_node: community_id}.
        merged_df: Full merged transaction-label DataFrame.
        phase_label: 'T2' or 'T3'.
        method_name: String name of the detection method.
    """
    scenario = "t2_coordinated_fraud" if phase_label == "T2" else "t3_mutated_fraud"
    fraud_acct_ids = merged_df[merged_df["scenario"] == scenario]["account_id"].unique()
    fraud_acct_nodes = [f"account:{aid}" for aid in fraud_acct_ids]

    community_assignments = [
        community_id_map.get(n) for n in fraud_acct_nodes if n in community_id_map
    ]

    if not community_assignments:
        print(f"  ⚠ [{method_name.upper()} / {phase_label}] No fraud ring accounts found in community map.")
        return

    unique_communities = set(a for a in community_assignments if a is not None and a >= 0)
    dominant_cid = max(unique_communities, key=lambda c: community_assignments.count(c))
    accounts_in_dominant = community_assignments.count(dominant_cid)

    print(f"\n  [{method_name.upper()} / {phase_label}] Fraud Ring Community Analysis:")
    print(f"    - {len(fraud_acct_ids)} fraud accounts span {len(unique_communities)} community/communities")
    print(f"    - Dominant community: #{dominant_cid} ({accounts_in_dominant}/{len(fraud_acct_ids)} ring accounts)")

    if len(unique_communities) == 1:
        print(f"    ✓ All 10 fraud accounts are in a SINGLE community. Ring perfectly detected!")
    elif accounts_in_dominant >= 8:
        print(f"    ✓ {accounts_in_dominant}/10 fraud accounts co-cluster. Strong ring signal.")
    else:
        print(
            f"    ⚠ Only {accounts_in_dominant}/10 ring accounts in dominant community. "
            f"Partial co-clustering (expected in {phase_label} if ring is diffuse)."
        )


# ---------------------------------------------------------------------------
# 7. Save Results
# ---------------------------------------------------------------------------

def save_community_results(phase_outputs, data_dir):
    """Save community assignments, statistics, and account features to data/.

    Args:
        phase_outputs: dict of {phase_label: {'community_maps', 'community_stats', 'account_features'}}.
        data_dir: Output directory path.
    """
    for phase_label, outputs in phase_outputs.items():
        label = phase_label.lower()

        # Community assignments (louvain only, as JSON)
        for method, id_map in outputs["community_maps"].items():
            assignment_path = os.path.join(data_dir, f"community_assignments_{label}_{method}.json")
            # Convert node keys to strings (already strings)
            with open(assignment_path, "w") as f:
                json.dump(id_map, f)
            print(f"  Saved {assignment_path}")

        # Community statistics per algorithm
        for method, stats_df in outputs["community_stats"].items():
            stats_path = os.path.join(data_dir, f"community_stats_{label}_{method}.csv")
            stats_df.to_csv(stats_path, index=False)
            print(f"  Saved {stats_path}")

        # Account features
        acct_feat_path = os.path.join(data_dir, f"account_features_{label}.csv")
        outputs["account_features"].to_csv(acct_feat_path, index=False)
        print(f"  Saved {acct_feat_path}")


# ---------------------------------------------------------------------------
# 8. Main Pipeline
# ---------------------------------------------------------------------------

def main():
    G_t1, G_t2, G_t3, merged_df, data_dir = load_phase_graphs()

    # Determine phase time boundaries (same logic as build_graph.py)
    min_time = merged_df["timestamp_dt"].min()
    max_time = merged_df["timestamp_dt"].max()
    phase_duration = (max_time - min_time) / 3
    t1_end = min_time + phase_duration
    t2_end = min_time + 2 * phase_duration

    phase_configs = [
        ("T1", G_t1, merged_df[merged_df["timestamp_dt"] < t1_end]),
        ("T2", G_t2, merged_df[(merged_df["timestamp_dt"] >= t1_end) & (merged_df["timestamp_dt"] < t2_end)]),
        ("T3", G_t3, merged_df[merged_df["timestamp_dt"] >= t2_end]),
    ]

    METHODS = ("louvain", "lpa")
    phase_outputs = {}

    for phase_label, G_hetero, phase_df in phase_configs:
        print(f"\n{'='*55}")
        print(f"  Phase {phase_label}: {len(phase_df)} transactions")
        print(f"{'='*55}")

        # Step 1: Project heterogeneous graph → account-account graph
        G_proj = project_to_account_graph(G_hetero)

        # Step 2: Community detection
        community_maps = detect_communities(G_proj, methods=METHODS, seed=42)

        # Step 3: Topological proof (only meaningful for T2 and T3)
        if phase_label in ("T2", "T3"):
            for method, id_map in community_maps.items():
                verify_ring_community(id_map, merged_df, phase_label, method)

        # Step 4: Community statistics per method
        community_stats_by_method = {}
        for method, id_map in community_maps.items():
            print(f"\n  Computing community stats [{method.upper()}]...")
            stats_df = compute_community_stats(id_map, G_hetero, G_proj, phase_df, phase_label)
            community_stats_by_method[method] = stats_df

            # Print top communities by size
            top = stats_df[stats_df["size"] >= 5].head(10)
            if not top.empty:
                print(top[[
                    "community_id", "size", "fraud_ratio", "density",
                    "shared_device_ratio", "shared_ip_ratio"
                ]].to_string(index=False))

        # Step 5: Account-level features
        print(f"\n  Computing account features [{phase_label}]...")
        account_features = compute_account_features(
            G_hetero, G_proj, community_maps, phase_df, phase_label
        )
        fraud_accts = account_features[account_features["is_fraud_account"]]
        print(f"  Fraud account feature sample ({len(fraud_accts)} fraud accounts):")
        if not fraud_accts.empty:
            print(fraud_accts[[
                "account_id", "degree", "connected_device_count",
                "connected_ip_count", "connected_merchant_count",
                "shared_device_count", "proj_degree", "community_id_louvain"
            ]].to_string(index=False))

        phase_outputs[phase_label] = {
            "community_maps": community_maps,
            "community_stats": community_stats_by_method,
            "account_features": account_features,
        }

    # Step 6: Save all results
    print("\n\nSaving community outputs...")
    save_community_results(phase_outputs, data_dir)
    print("\nDone.")


if __name__ == "__main__":
    main()
