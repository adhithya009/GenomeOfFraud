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

# Time periods (30 days total, split into three 10‑day phases)
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
    # Accounts
    accounts = []
    for _ in range(NUM_ACCOUNTS):
        created = fake.date_time_between(start_date="-90d", end_date="now")
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
        created = fake.date_time_between(start_date="-90d", end_date="now")
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
    transactions = []
    for _ in range(NUM_TRANSACTIONS):
        phase = random.choice(list(PHASES.values()))
        ts = random_date(*phase)
        transaction = {
            "transaction_id": str(uuid.uuid4()),
            "timestamp": ts.isoformat(),
            "account_id": accounts_df.sample(1).iloc[0]["account_id"],
            "merchant_id": merchants_df.sample(1).iloc[0]["merchant_id"],
            "device_id": devices_df.sample(1).iloc[0]["device_id"],
            "ip_id": ips_df.sample(1).iloc[0]["ip_id"],
            "amount": round(random.uniform(1, 5000), 2),
            "is_fraud": False,
        }
        transactions.append(transaction)
    return pd.DataFrame(transactions)

def inject_coordinated_fraud(transactions_df, accounts_df, merchants_df, devices_df, ips_df, phase_key, mutated=False):
    """Inject a coordinated fraud cluster.

    Args:
        transactions_df: existing transactions DataFrame to which fraud rows will be appended.
        phase_key: "T2" or "T3" indicating which time window to use.
        mutated: If True, each fraudulent account uses a distinct device (mutation).
    """
    start, end = PHASES[phase_key]
    # Choose a fraudulent merchant (newly created for clarity)
    fraud_merchant = merchants_df.sample(1).iloc[0]
    # Choose a set of accounts (e.g., 10) that are newly created near the start of the phase
    fraud_accounts = accounts_df.sort_values("created_at").head(10).copy()
    # Shared devices and IPs (unless mutated)
    shared_devices = devices_df.sample(3).copy()
    shared_ips = ips_df.sample(2).copy()

    for acct in fraud_accounts.itertuples():
        if mutated:
            device_id = devices_df[~devices_df["device_id"].isin(shared_devices["device_id"])].sample(1).iloc[0]["device_id"]
        else:
            device_id = shared_devices.sample(1).iloc[0]["device_id"]
        ip_id = shared_ips.sample(1).iloc[0]["ip_id"]
        for _ in range(random.randint(5, 15)):
            ts = random_date(start, end)
            transaction = {
                "transaction_id": str(uuid.uuid4()),
                "timestamp": ts.isoformat(),
                "account_id": acct.account_id,
                "merchant_id": fraud_merchant.merchant_id,
                "device_id": device_id,
                "ip_id": ip_id,
                "amount": round(random.uniform(100, 2000), 2),
                "is_fraud": True,
            }
            transactions_df = pd.concat([transactions_df, pd.DataFrame([transaction])], ignore_index=True)
    return transactions_df

def main(output_dir="data"):
    os.makedirs(output_dir, exist_ok=True)
    accounts_df, merchants_df, devices_df, ips_df = generate_entities()
    accounts_df.to_csv(os.path.join(output_dir, "accounts.csv"), index=False)
    merchants_df.to_csv(os.path.join(output_dir, "merchants.csv"), index=False)
    devices_df.to_csv(os.path.join(output_dir, "devices.csv"), index=False)
    ips_df.to_csv(os.path.join(output_dir, "ips.csv"), index=False)

    # Normal transactions
    transactions_df = generate_normal_transactions(accounts_df, merchants_df, devices_df, ips_df)

    # Inject known coordinated fraud in T2
    transactions_df = inject_coordinated_fraud(transactions_df, accounts_df, merchants_df, devices_df, ips_df, "T2", mutated=False)
    # Inject mutated coordinated fraud in T3 (device mutation)
    transactions_df = inject_coordinated_fraud(transactions_df, accounts_df, merchants_df, devices_df, ips_df, "T3", mutated=True)

    # Shuffle final transactions
    transactions_df = transactions_df.sample(frac=1).reset_index(drop=True)
    transactions_df.to_csv(os.path.join(output_dir, "transactions.csv"), index=False)

    print(f"Synthetic data generated in '{output_dir}'")

if __name__ == "__main__":
    main()
