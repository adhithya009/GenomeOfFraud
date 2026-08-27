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
│   ├── genome_reference_t1.json    # Causal T1 historical reference statistics
│   ├── genome_reference_t1_t2.json # Causal T1+T2 historical reference statistics
│   ├── genome_drift_report.csv     # Population distribution drift analysis (T2 -> T3)
│   ├── genome_drift_summary.json   # Human-readable drift summary & diagnostic metrics
│   └── model_results.json # Full benchmark results and 7-part ablation study metrics
├── src/                   # Python source code
│   ├── __init__.py
│   ├── generate_data.py   # Synthetic data generator, fraud injector, & validator
│   ├── build_graph.py     # Graph constructor & temporal decay engine
│   ├── community.py       # Projection graph generator & Louvain community detector
│   ├── features.py        # Behavioral gene extraction & global normalization
│   ├── genome_drift.py    # Historical reference engine, account drift, & relational anomaly detector
│   └── model.py           # Leakage-safe model trainer, out-of-time ablation evaluator, & risk layer
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

---

## Architecture: Causal Historical Reference & Dynamic Risk Layer

```
                Historical Genome Reference (T<P)
                               │
                               ▼
                        Current Account
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
    Genome Drift      Relational Anomaly     Behavioral Drift
   (Z-score vs Ref)   (Graph Topology)      (Surface Activity)
          │                    │                    │
          └────────────────────┼────────────────────┘
                               ▼
                       Hybrid Risk Layer
           (Supervised Prob + Relational + Drift)
                               │
                               ▼
                    Dynamic Risk Threshold
              (Historical Percentile Calibration)
                               │
                      ┌────────┴────────┐
                      ▼                 ▼
                   NORMAL             FRAUD
```

### Concept & Objectives

Genome drift occurs when fraud syndicates intentionally alter their operational surface tactics between time periods to evade traditional rule engines and supervised models.

The framework decomposes the fraud representation into two structural levels:
* **Level A: Individual Behavioral Genes**: Account velocity, device usage counts, transaction amounts, individual IP/merchant interaction frequencies. (Highly volatile during mutation)
* **Level B: Relational Genome**: Relational topology across shared entity infrastructure (`account ↔ IP`, `account ↔ merchant`, graph projection degree, community density). (Relatively stable during mutation)

### Temporal Reference Policy & Safeguards

To prevent future lookahead leakage ($T_1+T_2 \rightarrow T_3$):
* **Phase $T_1$ Reference**: Population baseline constructed from $T_1$.
* **Phase $T_2$ Reference**: Population baseline constructed strictly from $T_1$.
* **Phase $T_3$ Reference**: Population baseline constructed strictly from $T_1 + T_2$.

All references are saved to `data/genome_reference_*.json` and unit-tested for invariance against future data corruption.

---

## Out-of-Time 7-Part Ablation Study ($T_1+T_2 \rightarrow T_3$)

### Ranking Metrics Table (Ranking Quality across $T_3$)

| Experiment | PR-AUC | ROC-AUC | Features | Evaluation Mode |
| :--- | ---: | ---: | ---: | :--- |
| **Exp A: Temporal Baseline** | 0.1290 | 0.7533 | 25 | Supervised Model |
| **Exp B: Temporal FraudGenome** | 0.1377 | 0.7573 | 45 | Supervised Model |
| **Exp C: FraudGenome + Mutation-Aware** | 0.1140 | 0.5959 | 52 | Supervised Model |
| **Exp D: Relational-Only Mutation** | 0.2444 | 0.8388 | 23 | Supervised Model |
| **Exp E: Historical Genome Drift** | 0.1137 | 0.7279 | 8 | Supervised Model |
| **Exp F: Relational Anomaly Score** | **0.0939** | **0.8939** | 24 | Unsupervised Relational Anomaly |
| **Exp G: Hybrid Risk Layer** | **0.2293** | **0.8876** | 61 | Supervised + Relational + Drift |

### Decision Metrics Table (Deployed Detection Performance)

| Experiment | Precision | Recall | F1 | False Positives | False Negatives | FPR | Cutoff Threshold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Exp A: Temporal Baseline** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.05 (Fixed) |
| **Exp B: Temporal FraudGenome** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.05 (Fixed) |
| **Exp C: FraudGenome + Mutation-Aware** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.05 (Fixed) |
| **Exp D: Relational-Only Mutation** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.05 (Fixed) |
| **Exp E: Historical Genome Drift** | 0.2500 | 0.1000 | 0.1429 | 3 | 9 | 0.00151 | 0.05 (Fixed) |
| **Exp F: Relational Anomaly Score** | **0.0658** | **0.5000** | **0.1163** | 71 | 5 | 0.03568 | **1.06 (Dynamic 97.5th %ile)** |
| **Exp G: Hybrid Risk Layer** | **0.0725** | **0.5000** | **0.1266** | 64 | 5 | 0.03216 | **0.30 (Dynamic 97.5th %ile)** |

**Key Finding**:
* **Relational Anomaly Scoring (Exp F)** produces a massive ranking capability (**ROC-AUC 0.8939**).
* **Dynamic Threshold Calibration** fit on historical normal percentiles ($T_1+T_2$) increases $T_3$ fraud detection rate **5x from 10% (1/10) to 50% (5/10 fraud accounts detected)**.

---

## Setup & Usage

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline

To execute the complete pipeline and temporal unit tests:

```bash
python3 src/generate_data.py
python3 src/build_graph.py
python3 src/community.py
python3 src/features.py
python3 src/genome_drift.py
python3 src/model.py
```
