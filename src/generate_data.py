import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker()

# Configuration
NUM_ACCOUNTS = 2000
NUM_MERCHANTS = 200
NUM_DEVICES = 500
NUM_IPS = 1000
NUM_TRANSACTIONS = 20000

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

# Time periods (30 days total, split into three 10-day phases)
BASE_DATE = datetime.now() - timedelta(days=30)
PHASE_DAYS = 10
PHASES = {
    "T1": (BASE_DATE, BASE_DATE + timedelta(days=PHASE_DAYS)),
    "T2": (BASE_DATE + timedelta(days=PHASE_DAYS), BASE_DATE + timedelta(days=2 * PHASE_DAYS)),
    "T3": (BASE_DATE + timedelta(days=2 * PHASE_DAYS), BASE_DATE + timedelta(days=3 * PHASE_DAYS)),
}


def random_date(start, end):
    """Return a random datetime between ``start`` and ``end``."""
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return start + timedelta(seconds=random_seconds)


def generate_entities():
    # Accounts (created 10 to 90 days prior to BASE_DATE)
    accounts = []
    for _ in range(NUM_ACCOUNTS):
        created = fake.date_time_between(
            start_date=BASE_DATE - timedelta(days=90), end_date=BASE_DATE - timedelta(days=10)
        )
        accounts.append({
            "account_id": str(uuid.uuid4()),
            "created_at": created.isoformat(),
            "country": fake.country_code(),
            "kyc_status": random.choice(["verified", "unverified"]),
        })
    accounts_df = pd.DataFrame(accounts)

    # Merchants
    merchants = []
    for _ in range(NUM_MERCHANTS):
        created = fake.date_time_between(
            start_date=BASE_DATE - timedelta(days=90), end_date=BASE_DATE
        )
        merchants.append({
            "merchant_id": str(uuid.uuid4()),
            "created_at": created.isoformat(),
            "category": random.choice(["retail", "food", "travel", "digital"]),
        })
    merchants_df = pd.DataFrame(merchants)

    # Devices
    devices = []
    for _ in range(NUM_DEVICES):
        devices.append({
            "device_id": str(uuid.uuid4()),
            "device_type": random.choice(["mobile", "desktop", "tablet"]),
        })
    devices_df = pd.DataFrame(devices)

    # IPs
    ips = []
    for _ in range(NUM_IPS):
        ips.append({
            "ip_id": str(uuid.uuid4()),
            "ip_address": fake.ipv4(),
        })
    ips_df = pd.DataFrame(ips)

    return accounts_df, merchants_df, devices_df, ips_df


def generate_normal_transactions(accounts_df, merchants_df, devices_df, ips_df):
    """Generate normal transactions with realistic customer entity preference distributions."""
    transactions = []
    labels = []

    account_ids = accounts_df["account_id"].tolist()
    device_ids = devices_df["device_id"].tolist()
    ip_ids = ips_df["ip_id"].tolist()
    merchant_ids = merchants_df["merchant_id"].tolist()

    # Assign primary device, IP, and merchant preferences to normal accounts
    acct_preferences = {}
    for aid in account_ids:
        acct_preferences[aid] = {
            "devices": random.sample(device_ids, k=random.choice([1, 1, 2])),
            "ips": random.sample(ip_ids, k=random.choice([1, 2])),
            "merchants": random.sample(merchant_ids, k=random.choice([1, 2, 3])),
        }

    tx_per_phase = NUM_TRANSACTIONS // 3
    phase_scenarios = {
        "T1": "baseline_normal",
        "T2": "t2_normal",
        "T3": "t3_normal",
    }

    for phase_key, (start, end) in PHASES.items():
        scenario = phase_scenarios[phase_key]
        for _ in range(tx_per_phase):
            tx_id = str(uuid.uuid4())
            ts = random_date(start, end)
            acct_id = random.choice(account_ids)
            pref = acct_preferences[acct_id]

            # 85% of time use preferred merchant, 15% random
            if random.random() < 0.85:
                merch_id = random.choice(pref["merchants"])
            else:
                merch_id = random.choice(merchant_ids)

            # 90% of time use preferred device & IP
            dev_id = random.choice(pref["devices"]) if random.random() < 0.90 else random.choice(device_ids)
            ip_id = random.choice(pref["ips"]) if random.random() < 0.90 else random.choice(ip_ids)

            tx = {
                "transaction_id": tx_id,
                "timestamp": ts.isoformat(),
                "account_id": acct_id,
                "merchant_id": merch_id,
                "device_id": dev_id,
                "ip_id": ip_id,
                "amount": round(random.uniform(1, 5000), 2),
            }
            lbl = {
                "transaction_id": tx_id,
                "scenario": scenario,
                "is_fraud": False,
                "fraud_family": "none",
                "fraud_cluster_id": "none",
            }
            transactions.append(tx)
            labels.append(lbl)

    return transactions, labels


def inject_evolving_fraud_ring(
    transactions, labels, accounts_df, merchants_df, devices_df, ips_df
):
    """Inject 5 INDEPENDENT persistent fraud rings evolving from T2 (coordinated) to T3 (mutated).

    Each cluster k in {1..5} has its OWN 100% independent infrastructure:
    - 1 distinct merchant per cluster (5 merchants total, zero cross-cluster merchant sharing)
    - 2 distinct accounts per cluster (10 accounts total, zero cross-cluster account sharing)
    - 2 distinct shared IPs per cluster (10 IPs total, zero cross-cluster IP sharing)
    - T2: 1 shared device per cluster in T2 (5 T2 devices total, zero cross-cluster device sharing)
    - T3: 2 distinct devices per cluster in T3 (1 per account = 10 T3 devices total, zero cross-cluster device sharing)
    """
    t2_start, t2_end = PHASES["T2"]
    t3_start, t3_end = PHASES["T3"]

    NUM_CLUSTERS = 5
    ACCTS_PER_CLUSTER = 2

    # Select 10 fraud accounts total
    fraud_account_indices = accounts_df.sample(NUM_CLUSTERS * ACCTS_PER_CLUSTER, random_state=42).index
    fraud_accounts = accounts_df.loc[fraud_account_indices, "account_id"].tolist()

    # Select 5 distinct merchants (1 per cluster)
    fraud_merchants = merchants_df.sample(NUM_CLUSTERS, random_state=42)["merchant_id"].tolist()

    # Select 10 distinct IPs (2 per cluster)
    fraud_ips = ips_df.sample(NUM_CLUSTERS * 2, random_state=42)["ip_id"].tolist()

    # Select 5 distinct T2 shared devices (1 per cluster)
    t2_devices = devices_df.sample(NUM_CLUSTERS, random_state=42)["device_id"].tolist()

    # Select 10 distinct T3 mutated devices (1 per account)
    available_devices = devices_df[~devices_df["device_id"].isin(t2_devices)]
    t3_devices = available_devices.sample(NUM_CLUSTERS * ACCTS_PER_CLUSTER, random_state=43)["device_id"].tolist()

    fraud_family = "coordinated_ring"

    for c_idx in range(NUM_CLUSTERS):
        cluster_id = f"cluster_{c_idx + 1:03d}"

        # Disjoint infrastructure assigned to cluster c_idx
        c_accounts = fraud_accounts[c_idx * ACCTS_PER_CLUSTER : (c_idx + 1) * ACCTS_PER_CLUSTER]
        c_merchant = fraud_merchants[c_idx]
        c_ips = fraud_ips[c_idx * 2 : (c_idx + 1) * 2]
        c_t2_device = t2_devices[c_idx]
        c_t3_devices = t3_devices[c_idx * ACCTS_PER_CLUSTER : (c_idx + 1) * ACCTS_PER_CLUSTER]

        # Inject T2 Coordinated Fraud (elevated velocity + shared cluster T2 device + shared cluster IPs + shared cluster merchant)
        for acct_id in c_accounts:
            num_tx = random.randint(10, 20)  # Elevated activity in T2
            for _ in range(num_tx):
                tx_id = str(uuid.uuid4())
                ts = random_date(t2_start, t2_end)
                ip_id = random.choice(c_ips)

                tx = {
                    "transaction_id": tx_id,
                    "timestamp": ts.isoformat(),
                    "account_id": acct_id,
                    "merchant_id": c_merchant,
                    "device_id": c_t2_device,
                    "ip_id": ip_id,
                    "amount": round(random.uniform(500, 3000), 2),
                }
                lbl = {
                    "transaction_id": tx_id,
                    "scenario": "t2_coordinated_fraud",
                    "is_fraud": True,
                    "fraud_family": fraud_family,
                    "fraud_cluster_id": cluster_id,
                }
                transactions.append(tx)
                labels.append(lbl)

        # Inject T3 Mutated Fraud (camouflaged surface velocity + unique T3 device per account + retained cluster IPs & merchant)
        for a_idx, acct_id in enumerate(c_accounts):
            num_tx = random.randint(2, 5)  # Normal transaction count in T3 (~0.3 tx/day)
            c_t3_dev = c_t3_devices[a_idx]
            for _ in range(num_tx):
                tx_id = str(uuid.uuid4())
                ts = random_date(t3_start, t3_end)
                ip_id = random.choice(c_ips)

                tx = {
                    "transaction_id": tx_id,
                    "timestamp": ts.isoformat(),
                    "account_id": acct_id,
                    "merchant_id": c_merchant,
                    "device_id": c_t3_dev,
                    "ip_id": ip_id,
                    "amount": round(random.uniform(1, 5000), 2),
                }
                lbl = {
                    "transaction_id": tx_id,
                    "scenario": "t3_mutated_fraud",
                    "is_fraud": True,
                    "fraud_family": fraud_family,
                    "fraud_cluster_id": cluster_id,
                }
                transactions.append(tx)
                labels.append(lbl)

    return transactions, labels, accounts_df


def validate_dataset(transactions_df, labels_df, accounts_df, merchants_df, devices_df, ips_df):
    """Run automated checks to verify dataset integrity, topology constraints, and cluster independence."""
    print("\n--- Running Dataset Validation Checks ---")
    merged = pd.merge(transactions_df, labels_df, on="transaction_id")
    merged["dt"] = pd.to_datetime(merged["timestamp"])

    t1_start, t1_end = PHASES["T1"]

    # 1. T1 contains no injected fraud
    t1_tx = merged[(merged["dt"] >= t1_start) & (merged["dt"] < t1_end)]
    assert (
        t1_tx["is_fraud"].sum() == 0
    ), f"Validation Failed: T1 contains {t1_tx['is_fraud'].sum()} fraud transactions!"
    assert (
        (t1_tx["scenario"] == "baseline_normal").all()
    ), "Validation Failed: T1 contains non-baseline_normal scenarios!"
    print("✓ Check 1 Passed: T1 is 100% baseline normal (0% fraud).")

    # 2. Fraud scenarios exist
    t2_fraud = merged[merged["scenario"] == "t2_coordinated_fraud"]
    t3_fraud = merged[merged["scenario"] == "t3_mutated_fraud"]
    assert len(t2_fraud) > 0, "Validation Failed: T2 contains no coordinated fraud!"
    assert len(t3_fraud) > 0, "Validation Failed: T3 contains no mutated fraud!"
    print(f"✓ Check 2 Passed: T2 contains {len(t2_fraud)} tx, T3 contains {len(t3_fraud)} tx.")

    # 3. Exactly 5 distinct fraud clusters
    fraud_df = merged[merged["is_fraud"]]
    unique_clusters = sorted([c for c in fraud_df["fraud_cluster_id"].unique() if c != "none"])
    assert len(unique_clusters) == 5, f"Expected 5 fraud clusters, found {len(unique_clusters)}"
    print(f"✓ Check 3 Passed: 5 distinct fraud clusters present: {unique_clusters}.")

    # 4. Audit Cluster Independence (Zero shared infrastructure across clusters)
    cluster_accounts = fraud_df.groupby("fraud_cluster_id")["account_id"].unique().to_dict()
    cluster_merchants = fraud_df.groupby("fraud_cluster_id")["merchant_id"].unique().to_dict()
    cluster_ips = fraud_df.groupby("fraud_cluster_id")["ip_id"].unique().to_dict()
    cluster_t2_devs = t2_fraud.groupby("fraud_cluster_id")["device_id"].unique().to_dict()
    cluster_t3_devs = t3_fraud.groupby("fraud_cluster_id")["device_id"].unique().to_dict()

    for i in range(len(unique_clusters)):
        for j in range(i + 1, len(unique_clusters)):
            c1, c2 = unique_clusters[i], unique_clusters[j]
            acct_overlap = set(cluster_accounts[c1]).intersection(set(cluster_accounts[c2]))
            merch_overlap = set(cluster_merchants[c1]).intersection(set(cluster_merchants[c2]))
            ip_overlap = set(cluster_ips[c1]).intersection(set(cluster_ips[c2]))

            assert len(acct_overlap) == 0, f"Account overlap between {c1} and {c2}: {acct_overlap}"
            assert len(merch_overlap) == 0, f"Merchant overlap between {c1} and {c2}: {merch_overlap}"
            assert len(ip_overlap) == 0, f"IP overlap between {c1} and {c2}: {ip_overlap}"

    print("✓ Check 4 Passed: 100% Cluster Independence verified (zero shared accounts, merchants, or IPs between clusters).")

    # 5. Per-cluster internal topology verification (T2 sharing -> T3 mutation)
    for c_id in unique_clusters:
        c_t2 = t2_fraud[t2_fraud["fraud_cluster_id"] == c_id]
        c_t3 = t3_fraud[t3_fraud["fraud_cluster_id"] == c_id]

        c_t2_accts = c_t2["account_id"].nunique()
        c_t3_accts = c_t3["account_id"].nunique()
        assert c_t2_accts == 2 and c_t3_accts == 2, f"Cluster {c_id} expected 2 accounts per phase"

        c_t2_devs = c_t2["device_id"].nunique()
        c_t3_devs = c_t3["device_id"].nunique()
        assert c_t2_devs == 1, f"Cluster {c_id} T2 expected 1 shared device, got {c_t2_devs}"
        assert c_t3_devs == 2, f"Cluster {c_id} T3 expected 2 distinct devices, got {c_t3_devs}"

        c_t2_merch = set(c_t2["merchant_id"])
        c_t3_merch = set(c_t3["merchant_id"])
        assert len(c_t2_merch) == 1 and c_t2_merch == c_t3_merch, f"Cluster {c_id} merchant mismatch"

        c_t2_ips = set(c_t2["ip_id"])
        c_t3_ips = set(c_t3["ip_id"])
        assert len(c_t2_ips) == 2 and c_t2_ips == c_t3_ips, f"Cluster {c_id} IP mismatch"

    print("✓ Check 5 Passed: All 5 clusters independently satisfy T2 sharing and T3 mutation topology.")
    print("--- All Validation Checks Passed Successfully! ---\n")


def print_validation_summary(transactions_df, labels_df, accounts_df):
    """Print statistical summary table comparing normal and fraud populations."""
    merged = pd.merge(transactions_df, labels_df, on="transaction_id")
    merged["dt"] = pd.to_datetime(merged["timestamp"])
    accounts_df["created_at_dt"] = pd.to_datetime(accounts_df["created_at"])

    t1_start, t1_end = PHASES["T1"]
    t2_start, t2_end = PHASES["T2"]
    t3_start, t3_end = PHASES["T3"]

    def compute_stats(df_sub, phase_start, label_name):
        if df_sub.empty:
            return {}

        n_tx = len(df_sub)
        n_accts = df_sub["account_id"].nunique()

        tx_counts = df_sub.groupby("account_id").size()
        mean_tx_count = tx_counts.mean()
        velocity = mean_tx_count / PHASE_DAYS

        mean_amt = df_sub["amount"].mean()
        std_amt = df_sub["amount"].std()

        acct_sub = accounts_df[accounts_df["account_id"].isin(df_sub["account_id"])]
        ages = (phase_start - acct_sub["created_at_dt"]).dt.days
        mean_age = ages.mean()

        devs_per_acct = df_sub.groupby("account_id")["device_id"].nunique().mean()
        ips_per_acct = df_sub.groupby("account_id")["ip_id"].nunique().mean()

        total_devs = df_sub["device_id"].nunique()
        dev_acct_counts = df_sub.groupby("device_id")["account_id"].nunique()
        shared_devs = (dev_acct_counts >= 2).sum()
        dev_sharing_ratio = shared_devs / total_devs if total_devs > 0 else 0.0

        total_ips = df_sub["ip_id"].nunique()
        ip_acct_counts = df_sub.groupby("ip_id")["account_id"].nunique()
        shared_ips = (ip_acct_counts >= 2).sum()
        ip_sharing_ratio = shared_ips / total_ips if total_ips > 0 else 0.0

        top_merch_n = df_sub.groupby(["account_id", "merchant_id"]).size().groupby("account_id").max()
        merch_conc = (top_merch_n / tx_counts).mean()

        return {
            "Label": label_name,
            "Total Tx": n_tx,
            "Active Accts": n_accts,
            "Tx/Acct": round(mean_tx_count, 2),
            "Velocity (tx/day)": round(velocity, 2),
            "Avg Amount ($)": round(mean_amt, 2),
            "Std Amount ($)": round(std_amt, 2),
            "Account Age (days)": round(mean_age, 1),
            "Devs/Acct": round(devs_per_acct, 2),
            "IPs/Acct": round(ips_per_acct, 2),
            "Device Sharing Ratio": round(dev_sharing_ratio, 3),
            "IP Sharing Ratio": round(ip_sharing_ratio, 3),
            "Merch Conc": round(merch_conc, 3),
        }

    t1_norm_df = merged[(merged["dt"] >= t1_start) & (merged["dt"] < t1_end)]
    t2_fraud_df = merged[merged["scenario"] == "t2_coordinated_fraud"]
    t3_norm_df = merged[(merged["dt"] >= t3_start) & (~merged["is_fraud"])]
    t3_fraud_df = merged[merged["scenario"] == "t3_mutated_fraud"]

    s_t1_norm = compute_stats(t1_norm_df, t1_start, "T1 Normal")
    s_t2_fraud = compute_stats(t2_fraud_df, t2_start, "T2 Fraud (Coordinated)")
    s_t3_norm = compute_stats(t3_norm_df, t3_start, "T3 Normal")
    s_t3_fraud = compute_stats(t3_fraud_df, t3_start, "T3 Mutated Fraud")

    stats_df = pd.DataFrame([s_t1_norm, s_t2_fraud, s_t3_norm, s_t3_fraud])

    print("=" * 100)
    print("  BEHAVIORAL & TOPOLOGY VALIDATION SUMMARY")
    print("=" * 100)
    print(stats_df.to_string(index=False))
    print("=" * 100)


def main(output_dir=None):
    if output_dir is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_dir = os.path.join(project_root, "data")

    os.makedirs(output_dir, exist_ok=True)
    accounts_df, merchants_df, devices_df, ips_df = generate_entities()

    # Generate normal transactions
    transactions, labels = generate_normal_transactions(
        accounts_df, merchants_df, devices_df, ips_df
    )

    # Inject persistent, evolving fraud rings (T2 -> T3)
    transactions, labels, accounts_df = inject_evolving_fraud_ring(
        transactions, labels, accounts_df, merchants_df, devices_df, ips_df
    )

    # Convert to DataFrames
    transactions_df = pd.DataFrame(transactions)
    labels_df = pd.DataFrame(labels)

    # Shuffle transactions & maintain label alignment
    transactions_df = transactions_df.sample(frac=1, random_state=42).reset_index(drop=True)
    labels_df = labels_df.set_index("transaction_id").loc[transactions_df["transaction_id"]].reset_index()

    # Save to CSV files
    accounts_df.to_csv(os.path.join(output_dir, "accounts.csv"), index=False)
    merchants_df.to_csv(os.path.join(output_dir, "merchants.csv"), index=False)
    devices_df.to_csv(os.path.join(output_dir, "devices.csv"), index=False)
    ips_df.to_csv(os.path.join(output_dir, "ips.csv"), index=False)
    transactions_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)
    labels_df.to_csv(os.path.join(output_dir, "fraud_labels.csv"), index=False)

    print(f"Synthetic data & fraud labels generated successfully in '{output_dir}'.")

    # Run automated dataset topology validation & print validation summary
    validate_dataset(transactions_df, labels_df, accounts_df, merchants_df, devices_df, ips_df)
    print_validation_summary(transactions_df, labels_df, accounts_df)


if __name__ == "__main__":
    main()
