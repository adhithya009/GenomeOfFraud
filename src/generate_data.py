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
    # Accounts (Baseline normal accounts created prior to T1)
    accounts = []
    for _ in range(NUM_ACCOUNTS):
        created = fake.date_time_between(
            start_date=BASE_DATE - timedelta(days=90), end_date=BASE_DATE
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
    """Generate normal transactions distributed explicitly across T1, T2, and T3."""
    transactions = []
    labels = []

    # Distribute normal transactions evenly across T1, T2, T3
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
            tx = {
                "transaction_id": tx_id,
                "timestamp": ts.isoformat(),
                "account_id": accounts_df.sample(1).iloc[0]["account_id"],
                "merchant_id": merchants_df.sample(1).iloc[0]["merchant_id"],
                "device_id": devices_df.sample(1).iloc[0]["device_id"],
                "ip_id": ips_df.sample(1).iloc[0]["ip_id"],
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
    """Inject a persistent fraud ring that evolves from T2 (coordinated) to T3 (mutated).

    Features:
    - 1 shared merchant
    - 10 accounts created near the start of T2
    - 2 shared IPs across T2 & T3
    - T2: 3 shared devices across all 10 accounts
    - T3: 10 distinct devices (1 per account mutation) while keeping same 2 IPs & merchant
    """
    t2_start, t2_end = PHASES["T2"]
    t3_start, t3_end = PHASES["T3"]

    # Select 1 fraud merchant
    fraud_merchant_id = merchants_df.sample(1).iloc[0]["merchant_id"]

    # Select 10 fraud accounts and set created_at near start of T2 (within 48h prior to T2 start)
    fraud_account_indices = accounts_df.sample(10).index
    fraud_accounts = []
    for idx in fraud_account_indices:
        created_dt = t2_start - timedelta(hours=random.randint(1, 48))
        accounts_df.loc[idx, "created_at"] = created_dt.isoformat()
        fraud_accounts.append(accounts_df.loc[idx, "account_id"])

    # Shared IPs (2 IPs)
    shared_ips = ips_df.sample(2)["ip_id"].tolist()

    # T2 Shared Devices (3 devices)
    t2_devices = devices_df.sample(3)["device_id"].tolist()

    # T3 Mutated Devices (10 distinct devices, 1 per fraud account)
    available_devices = devices_df[~devices_df["device_id"].isin(t2_devices)]
    t3_devices = available_devices.sample(10)["device_id"].tolist()
    account_t3_device_map = dict(zip(fraud_accounts, t3_devices))

    cluster_id = "cluster_001"
    fraud_family = "coordinated_ring"

    # Inject T2 Coordinated Fraud
    for acct_id in fraud_accounts:
        num_tx = random.randint(5, 15)
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
                "amount": round(random.uniform(100, 2000), 2),
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

    # Inject T3 Mutated Fraud (device mutation per account)
    for acct_id in fraud_accounts:
        num_tx = random.randint(5, 15)
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
                "amount": round(random.uniform(100, 2000), 2),
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
    """Run automated checks to verify dataset integrity and experimental constraints."""
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

    # 5. T3 fraud uses ~10 devices (1 per account)
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

    # 8. T2/T3 retain IP structure
    t2_ips = set(t2_fraud["ip_id"])
    t3_ips = set(t3_fraud["ip_id"])
    assert (
        t2_ips == t3_ips and len(t2_ips) == 2
    ), f"Validation Failed: T2/T3 fraud IPs mismatch (T2: {t2_ips}, T3: {t3_ips})!"
    print("✓ Check 8 Passed: T2 and T3 fraud share the exact same 2 IP addresses.")

    # 9. Fraud accounts created near start of T2
    fraud_accts_df = accounts_df[accounts_df["account_id"].isin(t2_accounts)]
    fraud_created_dts = pd.to_datetime(fraud_accts_df["created_at"])
    for dt in fraud_created_dts:
        assert (
            t2_start - timedelta(hours=48) <= dt <= t2_start
        ), f"Validation Failed: Fraud account creation date {dt} is outside T2 window!"
    print("✓ Check 9 Passed: All 10 fraud accounts were created near the start of T2 (within 48h).")

    print("--- All Validation Checks Passed Successfully! ---\n")


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

    # Inject persistent, evolving fraud ring (T2 -> T3)
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

    # Run automated dataset validation
    validate_dataset(transactions_df, labels_df, accounts_df, merchants_df, devices_df, ips_df)


if __name__ == "__main__":
    main()
