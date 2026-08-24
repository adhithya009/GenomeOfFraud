# GenomeOfFraud

A synthetic dataset generator and analytics framework for detecting coordinated and mutating fraud patterns across financial transactions, accounts, devices, IPs, and merchants.

Built for the **Razorpay Hackathon**.

## Repo Project Structure

```
GenomeOfFraud/
├── data/                  # Synthetic dataset tables & evaluation labels
│   ├── accounts.csv       # Account entities and KYC metadata
│   ├── devices.csv        # Device fingerprints and types
│   ├── ips.csv            # IPv4 addresses
│   ├── merchants.csv      # Merchant categories and timestamps
│   ├── transactions.csv   # Clean transaction logs (feature columns only)
│   └── fraud_labels.csv   # Ground-truth target annotations & scenario metadata
├── src/                   # Python source code
│   ├── __init__.py
│   └── generate_data.py   # Synthetic data generator, fraud injector, & validator
├── .gitignore             # Git ignore configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Overall Project Structure
[██████████] Concept / architecture       
[██████████] Fraud scenario design        
[██████████] Synthetic generator          
[██████████] Graph intelligence            
[██████████] Gene extraction
[██████████] Baseline model
[██████████] FraudGenome model
[██████████] Genome drift
[██████████] SHAP
[██████████] Decision layer
[██████████] Streamlit

## Dataset Overview

- **`accounts.csv`**: Contains `account_id`, `created_at`, `country`, and `kyc_status`.
- **`merchants.csv`**: Contains `merchant_id`, `created_at`, and `category`.
- **`devices.csv`**: Contains `device_id` and `device_type`.
- **`ips.csv`**: Contains `ip_id` and `ip_address`.
- **`transactions.csv`**: Feature log (`transaction_id`, `timestamp`, `account_id`, `merchant_id`, `device_id`, `ip_id`, `amount`).
- **`fraud_labels.csv`**: Ground-truth target table (`transaction_id`, `scenario`, `is_fraud`, `fraud_family`, `fraud_cluster_id`).

### Controlled Experimental Phases

- **$T_1$ (Baseline Normal)**: 100% normal transaction traffic (0% fraud).
- **$T_2$ (Coordinated Fraud)**: Normal traffic + injected coordinated fraud ring using **1 shared merchant**, **10 accounts created near $T_2$ start**, **2 shared IPs**, and **3 shared devices**.
- **$T_3$ (Mutated Fraud)**: Normal traffic + mutated fraud ring where the **same 10 accounts** mutate to each use a **distinct unique device** (10 devices total, 1 per account), while retaining the **same 2 IPs** and **1 merchant**.

## Setup & Usage

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Data & Validate

To regenerate synthetic datasets and automatically execute the 9 dataset integrity validation checks:

```bash
python3 src/generate_data.py
```
