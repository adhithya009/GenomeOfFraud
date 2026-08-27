# GenomeOfFraud

A synthetic dataset generator and analytics framework for detecting coordinated and mutating fraud patterns across financial transactions, accounts, devices, IPs, and merchants.

Built for the **Razorpay Hackathon**.

## Repo Project Structure

```
GenomeOfFraud/
├── data/                  # Synthetic dataset tables, graph pickles, & evaluation results
│   ├── accounts.csv       # Account entities and KYC metadata
│   ├── devices.csv        # Device fingerprints and types
│   ├── ips.csv            # IPv4 addresses
│   ├── merchants.csv      # Merchant categories and timestamps
│   ├── transactions.csv   # Clean transaction logs (feature columns only)
│   ├── fraud_labels.csv   # Ground-truth target annotations & scenario metadata
│   ├── genome_full.csv    # Extracted behavioral & mutation-aware features
│   ├── genome_drift_report.csv # Feature distribution drift analysis (T2 -> T3)
│   ├── genome_drift_summary.json # Human-readable drift summary & diagnostic metrics
│   └── model_results.json # Full benchmark results and ablation study metrics
├── src/                   # Python source code
│   ├── __init__.py
│   ├── generate_data.py   # Synthetic data generator, fraud injector, & validator
│   ├── build_graph.py     # Graph constructor & temporal decay engine
│   ├── community.py       # Projection graph generator & Louvain community detector
│   ├── features.py        # Behavioral gene extraction & global normalization
│   ├── genome_drift.py    # Genome drift quantification & mutation-aware feature extractor
│   └── model.py           # Leakage-safe model trainer & out-of-time ablation evaluator
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

---

## Genome Drift Detection

### Concept & Objectives

Genome drift occurs when fraud syndicates intentionally alter their operational surface tactics between time periods to evade traditional rule engines and supervised models.

The framework decomposes the fraud representation into two structural levels:
* **Level A: Individual Behavioral Genes**: Account velocity, device usage counts, transaction amounts, individual IP/merchant interaction frequencies. (Highly volatile during mutation)
* **Level B: Relational Genome**: Relational topology across shared entity infrastructure (`account ↔ IP`, `account ↔ merchant`, graph projection degree, community density). (Relatively stable during mutation)

### Temporal Causality Safeguards

To ensure valid out-of-time evaluation ($T_1+T_2 \rightarrow T_3$):
1. **Zero Ground-Truth Leakage**: All drift scores and mutation features (`net_ip_persistence_score`, `merch_persistence_score`, `dev_mutation_score`, `relational_stability_score`, `topology_drift_score`, `behavioral_shift_score`, `genome_drift_score`) are computed without reference to target labels (`is_fraud`, `fraud_cluster_id`, `scenario`, or `graph_community_fraud_ratio`).
2. **Historical Reference Isolation**: Features for phase $P$ use historical references constructed strictly from preceding phases ($T_{<P}$).
3. **Validation Threshold Tuning**: Decision thresholds for $T_3$ evaluation are fit strictly on $T_1+T_2$ validation data, never on $T_3$ test set labels.

### Measured Out-of-Time Ablation Experiment ($T_1+T_2 \rightarrow T_3$)

| Experiment | PR-AUC | ROC-AUC | Precision | Recall | F1 | FP | FN | Threshold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Exp A: Temporal Baseline** | 0.1241 | 0.8080 | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.05 |
| **Exp B: Temporal FraudGenome** | 0.1377 | 0.7573 | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.05 |
| **Exp C: FraudGenome + Mutation-Aware** | 0.1140 | 0.5959 | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.05 |
| **Exp D: Relational-Only Mutation** | **0.2489** | **0.9221** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.05 |

**Key Finding**: Filtering down to **Relational-Only / Graph-Heavy Mutation features** (Exp D) strips away volatile surface behaviors (device changes & velocity drops) and boosts out-of-time ranking power significantly:
* **ROC-AUC** increases from **0.8080** to **0.9221** (+14.1% improvement).
* **PR-AUC** increases from **0.1241** to **0.2489** (+100.5% relative improvement).

---

## Setup & Usage

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline

To execute the complete pipeline cleanly:

```bash
python3 src/generate_data.py
python3 src/build_graph.py
python3 src/community.py
python3 src/features.py
python3 src/genome_drift.py
python3 src/model.py
```
