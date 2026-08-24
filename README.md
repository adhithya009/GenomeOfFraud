# GenomeOfFraud

A synthetic dataset generator and analytics framework for detecting coordinated and mutating fraud patterns across financial transactions, accounts, devices, IPs, and merchants.

Built for the **Razorpay Hackathon**.

## Project Structure

```
GenomeOfFraud/
├── data/                  # CSV datasets containing generated entities & transactions
│   ├── accounts.csv       # Account entities and KYC metadata
│   ├── devices.csv        # Device fingerprints and types
│   ├── ips.csv            # IPv4 addresses
│   ├── merchants.csv      # Merchant categories and timestamps
│   └── transactions.csv   # Financial transaction logs with fraud annotations
├── src/                   # Python source code
│   ├── __init__.py
│   └── generate_data.py   # Synthetic data generator & fraud injector
├── .gitignore             # Git ignore configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Dataset Overview

- **`accounts.csv`**: Contains `account_id`, `created_at`, `country`, and `kyc_status`.
- **`merchants.csv`**: Contains `merchant_id`, `created_at`, and `category`.
- **`devices.csv`**: Contains `device_id` and `device_type`.
- **`ips.csv`**: Contains `ip_id` and `ip_address`.
- **`transactions.csv`**: Transaction history spanning 30 days across three 10-day phases (T1, T2, T3), including injected baseline coordinated fraud (T2) and mutated coordinated fraud (T3).

## Setup & Usage

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Data

To regenerate synthetic datasets into the `data/` folder:

```bash
python3 src/generate_data.py
```
