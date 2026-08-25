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
    """Inject persistent fraud rings evolving from T2 (coordinated) to T3 (mutated).

    Structure:
    - 1 shared merchant across T2 & T3
    - 10 accounts divided into 5 distinct fraud clusters (cluster_001 to cluster_005, 2 accounts per cluster)
    - 2 shared IPs across T2 & T3
    - T2: 3 shared devices across fraud accounts + elevated velocity/amounts
    - T3: 10 distinct devices (1 per account) + normal velocity/amounts
    """
    t2_start, t2_end = PHASES["T2"]
    t3_start, t3_end = PHASES["T3"]

    # Select 1 fraud merchant
    fraud_merchant_id = merchants_df.sample(1, random_state=42).iloc[0]["merchant_id"]

    # Select 10 fraud accounts from accounts_df
    fraud_account_indices = accounts_df.sample(10, random_state=42).index
    fraud_accounts = accounts_df.loc[fraud_account_indices, "account_id"].tolist()

    # Assign 5 distinct fraud clusters (2 accounts per cluster)
    fraud_cluster_map = {}
    for i, acct_id in enumerate(fraud_accounts):
        cluster_num = (i // 2) + 1
        fraud_cluster_map[acct_id] = f"cluster_{cluster_num:03d}"

    # Shared IPs (2 IPs)
    shared_ips = ips_df.sample(2, random_state=42)["ip_id"].tolist()

    # T2 Shared Devices (3 devices)
    t2_devices = devices_df.sample(3, random_state=42)["device_id"].tolist()

    # T3 Mutated Devices (10 distinct devices, 1 per fraud account)
    available_devices = devices_df[~devices_df["device_id"].isin(t2_devices)]
    t3_devices = available_devices.sample(10, random_state=43)["device_id"].tolist()
    account_t3_device_map = dict(zip(fraud_accounts, t3_devices))

    fraud_family = "coordinated_ring"

    # Inject T2 Coordinated Fraud (elevated velocity + distinct amounts + shared devices/IPs/merchant)
    for acct_id in fraud_accounts:
        cluster_id = fraud_cluster_map[acct_id]
        num_tx = random.randint(10, 20)  # Elevated activity in T2
        for _ in range(num_tx):
            tx_id = str(uuid.uuid4())
            ts = random_date(t2_start, t2_end)
            device_id = random.choice(t2_devices)
            ip_id = random.choice(shared_ips)

            tx = {
                "transaction_id": tx_id,
                "timestamp": ts.isoformat(),
                "account_id": acct_id,
                "merchant_id": fraud_merchant_id,
                "device_id": device_id,
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

    # Inject T3 Mutated Fraud (camouflaged surface behavior + 10 distinct devices)
    for acct_id in fraud_accounts:
        cluster_id = fraud_cluster_map[acct_id]
        num_tx = random.randint(2, 5)  # Normal transaction count in T3 (~0.3 tx/day)
        device_id = account_t3_device_map[acct_id]  # Unique device for this account in T3
        for _ in range(num_tx):
            tx_id = str(uuid.uuid4())
            ts = random_date(t3_start, t3_end)
            ip_id = random.choice(shared_ips)  # Retain same 2 shared IPs

            tx = {
                "transaction_id": tx_id,
                "timestamp": ts.isoformat(),
                "account_id": acct_id,
                "merchant_id": fraud_merchant_id,
                "device_id": device_id,
                "ip_id": ip_id,
                "amount": round(random.uniform(1, 5000), 2),  # Uniform matching normal population
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
    """Run automated checks to verify dataset integrity and experimental topology constraints."""
    print("\n--- Running Dataset Validation Checks ---")
    merged = pd.merge(transactions_df, labels_df, on="transaction_id")
    merged["dt"] = pd.to_datetime(merged["timestamp"])

    t1_start, t1_end = PHASES["T1"]
    t2_start, t2_end = PHASES["T2"]
    t3_start, t3_end = PHASES["T3"]

    # 1. T1 contains no injected fraud
    t1_tx = merged[(merged["dt"] >= t1_start) & (merged["dt"] < t1_end)]
    assert (
        t1_tx["is_fraud"].sum() == 0
    ), f"Validation Failed: T1 contains {t1_tx['is_fraud'].sum()} fraud transactions!"
    assert (
        (t1_tx["scenario"] == "baseline_normal").all()
    ), "Validation Failed: T1 contains non-baseline_normal scenarios!"
    print("✓ Check 1 Passed: T1 is 100% baseline normal (0% fraud).")

    # 2. T2 contains coordinated fraud
    t2_fraud = merged[merged["scenario"] == "t2_coordinated_fraud"]
    assert len(t2_fraud) > 0, "Validation Failed: T2 contains no coordinated fraud!"
    print(f"✓ Check 2 Passed: T2 contains {len(t2_fraud)} coordinated fraud transactions.")

    # 3. T3 contains mutated fraud
    t3_fraud = merged[merged["scenario"] == "t3_mutated_fraud"]
    assert len(t3_fraud) > 0, "Validation Failed: T3 contains no mutated fraud!"
    print(f"✓ Check 3 Passed: T3 contains {len(t3_fraud)} mutated fraud transactions.")

    # 4. T2 fraud uses <= 3 devices
    t2_devices_count = t2_fraud["device_id"].nunique()
    assert (
        t2_devices_count <= 3
    ), f"Validation Failed: T2 fraud uses {t2_devices_count} devices (expected <= 3)!"
    print(f"✓ Check 4 Passed: T2 fraud uses {t2_devices_count} shared devices (<= 3).")

    # 5. T3 fraud uses 10 devices (1 per account)
    t3_devices_count = t3_fraud["device_id"].nunique()
    assert (
        t3_devices_count == 10
    ), f"Validation Failed: T3 fraud uses {t3_devices_count} devices (expected 10)!"
    print(f"✓ Check 5 Passed: T3 fraud uses {t3_devices_count} mutated devices (10 distinct).")

    # 6. T2/T3 share merchant
    t2_merchant = t2_fraud["merchant_id"].unique()
    t3_merchant = t3_fraud["merchant_id"].unique()
    assert (
        len(t2_merchant) == 1 and len(t3_merchant) == 1 and t2_merchant[0] == t3_merchant[0]
    ), "Validation Failed: T2 and T3 fraud do not share the exact same merchant!"
    print("✓ Check 6 Passed: T2 and T3 fraud share the exact same merchant.")

    # 7. T2/T3 share fraud accounts
    t2_accounts = set(t2_fraud["account_id"])
    t3_accounts = set(t3_fraud["account_id"])
    assert (
        t2_accounts == t3_accounts and len(t2_accounts) == 10
    ), "Validation Failed: T2 and T3 fraud accounts mismatch!"
    print("✓ Check 7 Passed: T2 and T3 fraud share the exact same 10 accounts.")

    # 8. Check 5 distinct fraud clusters
    unique_clusters = set(merged[merged["is_fraud"]]["fraud_cluster_id"].unique())
    assert len(unique_clusters) == 5, f"Expected 5 fraud clusters, found {len(unique_clusters)}"
    print(f"✓ Check 8 Passed: Found {len(unique_clusters)} distinct fraud clusters: {sorted(list(unique_clusters))}.")

    print("--- All Topology Checks Passed Successfully! ---\n")


def print_validation_summary(transactions_df, labels_df, accounts_df):
    """Print statistical comparison validating that T3 fraud surface behavior overlaps with normal behavior."""
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
