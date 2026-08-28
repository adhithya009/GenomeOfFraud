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
│   ├── risk_calibration_t1_t2.json # Frozen empirical CDF & policy calibration profile
│   ├── genome_drift_report.csv     # Population distribution drift analysis (T2 -> T3)
│   ├── genome_drift_summary.json   # Human-readable drift summary & diagnostic metrics
│   └── model_results.json # Full benchmark results, calibration, & ablation metrics
├── src/                   # Python source code
│   ├── __init__.py
│   ├── generate_data.py   # Synthetic data generator, fraud injector, & validator
│   ├── build_graph.py     # Graph constructor & temporal decay engine
│   ├── community.py       # Projection graph generator & Louvain community detector
│   ├── features.py        # Behavioral gene extraction & global normalization
│   ├── genome_drift.py    # Historical reference engine, ECDF calibration, & policy layer
│   ├── model.py           # Leakage-safe model trainer, out-of-time ablation, & alert evaluator
│   └── xai.py             # SHAP, gene group attribution, hybrid decomposition, & human-readable XAI
├── .gitignore             # Git ignore configuration
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

## Overall Project Progress

[██████████] Concept / architecture   
[██████████] Fraud scenario design  
[██████████] Synthetic generator  
[██████████] Graph intelligence  
[██████████] Gene extraction  
[██████████] Baseline model  
[██████████] FraudGenome model  
[██████████] Genome drift  
[██████████] Decision layer & Frozen Calibration  
[██████████] SHAP & Explainable AI (XAI)  
[██████████] Streamlit  

---

## Explainable AI (XAI) Architecture

```
             Final Risk Score
                    │
         ┌──────────┼──────────┐
         ▼          ▼          ▼
     Supervised  Relational   Genome
       Risk       Anomaly      Drift
         │          │          │
         ▼          ▼          ▼
       SHAP      Historical   Historical
     (Tree)     Deviation    Deviation
         │          │          │
         └──────────┼──────────┘
                    ▼
          Human Evidence Summary
                    │
                    ▼
             Policy Decision
```

### Multilayer Explanation Taxonomy

GenomeOfFraud answers **WHY** an account was assigned a specific risk level across six explanation layers:

1. **Supervised Model Attribution**: Quantifies predictive feature contributions to supervised probability using `shap.TreeExplainer`.
2. **Behavioral Gene-Group Attribution**: Aggregates SHAP contributions across seven gene categories (*Account*, *Device*, *Network*, *Merchant*, *Transaction*, *Graph*, *Community*).
3. **Relational Anomaly Explanation**: Decomposes infrastructure sharing score into constituent $+Z\sigma$ standardized elevations above historical baselines.
4. **Account Genome Drift Explanation**: Details individual feature and gene-group $Z$-score shifts relative to historical account reference profiles.
5. **Hybrid Risk Component Decomposition**: Decomposes final risk score into exact additive percentile contributions:
   $$\text{Risk}_{\text{final}} = w_1 \cdot \text{CDF}(\text{Supervised}) + w_2 \cdot \text{CDF}(\text{Relational}) + w_3 \cdot \text{CDF}(\text{Drift})$$
6. **Policy Decision Explanation**: Maps final hybrid risk score to cost-sensitive categories (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) and actions (`ALLOW`, `MONITOR_SOFT_CHALLENGE`, `STEP_UP_VERIFICATION`, `BLOCK_MANUAL_REVIEW`).

### Critical Scientific Disclaimers

* **SHAP $\neq$ Causality**: SHAP values describe statistical model feature attribution. They do NOT prove causal relationship or ground-truth intent.
* **SHAP $\neq$ Fraud Probability**: SHAP values represent log-odds margins or probability contributions to model output, not absolute fraud likelihood.
* **Genome Drift $\neq$ Fraud**: Statistical deviation from historical baseline describes behavioral shift. Legitimate user behavior may drift without representing fraud.
* **Relational Anomaly $\neq$ Fraud**: Infrastructure sharing (e.g. public Wi-Fi or multi-user device) indicates structural abnormality, not definitive malicious activity.

---

## Out-of-Time Experimental Evaluation ($T_1+T_2 \rightarrow T_3$)

### Ranking vs Decision Metrics

* **Ranking**: *"How well does the system order suspicious accounts?"* (PR-AUC, ROC-AUC)
* **Decision**: *"How many accounts are actually flagged at the chosen policy threshold?"* (Precision, Recall, F1, FP, FN, FPR)

---

### A. Ranking Comparison Table ($T_3$)

| Experiment | PR-AUC | ROC-AUC | Feature Count | Mode |
| :--- | ---: | ---: | ---: | :--- |
| **Exp A: Temporal Baseline** | 0.1290 | 0.7533 | 25 | Supervised Model |
| **Exp B: Temporal FraudGenome** | 0.1377 | 0.7573 | 45 | Supervised Model |
| **Exp C: FraudGenome + Mutation-Aware** | 0.1140 | 0.5959 | 52 | Supervised Model |
| **Exp D: Relational-Only Mutation** | **0.2462** | **0.8388** | 23 | Supervised Model |
| **Exp E: Historical Genome Drift** | 0.1168 | 0.5930 | 8 | Supervised Model |
| **Exp F: Relational Anomaly Score** | **0.0623** | **0.8648** | 24 | Unsupervised Relational Anomaly |
| **Exp G: Calibrated Hybrid Risk Layer** | **0.1114** | **0.8480** | 61 | Supervised + Relational + Drift |

---

### B. Decision Comparison Table ($T_3$)

| Experiment | Precision | Recall | F1 | FP | FN | FPR | Cutoff Threshold |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **Exp A: Temporal Baseline** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.0500 (Fixed) |
| **Exp B: Temporal FraudGenome** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.0500 (Fixed) |
| **Exp C: FraudGenome + Mutation-Aware** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.0500 (Fixed) |
| **Exp D: Relational-Only Mutation** | 1.0000 | 0.1000 | 0.1818 | 0 | 9 | 0.00000 | 0.0500 (Fixed) |
| **Exp E: Historical Genome Drift** | 0.2500 | 0.1000 | 0.1429 | 3 | 9 | 0.00151 | 0.0500 (Fixed) |
| **Exp F: Relational Anomaly Score** | **0.0225** | **0.6000** | **0.0433** | 261 | 4 | 0.13116 | **0.8938 (Frozen $p_{97.5}$)** |
| **Exp G: Calibrated Hybrid Risk Layer** | **0.1143** | **0.4000** | **0.1778** | 31 | 6 | 0.01558 | **0.8938 (Frozen $p_{97.5}$)** |

---

### C. Alert-Budget & Precision@K Evaluation Table ($T_3$)

In operational fraud teams, review capacity is constrained by analyst budgets:

| Model | Capacity Budget | Alerts Flagged | Fraud Captured | Total Fraud | Capture Rate (Recall) | Precision |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Calibrated Hybrid Risk** | Top 0.5% | 10 | 2 | 10 | 0.2000 (20%) | **0.2000 (20.0%)** |
| **Calibrated Hybrid Risk** | Top 1.0% | 20 | 3 | 10 | 0.3000 (30%) | **0.1500 (15.0%)** |
| **Calibrated Hybrid Risk** | Top 2.0% | 40 | 4 | 10 | 0.4000 (40%) | **0.1000 (10.0%)** |
| **Calibrated Hybrid Risk** | Top 5.0% | 100 | 5 | 10 | 0.5000 (50%) | **0.0500 (5.0%)** |

---

### D. Historical Hybrid Weighting Ablation ($T_1 \rightarrow T_2$ Validation Fold)

Weight selection is conducted strictly on historical $T_1+T_2$ validation data (never on $T_3$ labels):

| Weight Set Configuration | Weights $(w_1, w_2, w_3)$ | PR-AUC | ROC-AUC | Precision | Recall | F1 | Selection Status |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| **Set A (Baseline)** | $(0.4, 0.4, 0.2)$ | 0.5269 | 0.9909 | 0.0062 | 1.0000 | 0.0123 | Candidate |
| **Set B (Relational Heavy)** | $(0.3, 0.5, 0.2)$ | 0.4976 | 0.9896 | 0.0078 | 1.0000 | 0.0156 | Candidate |
| **Set C (Rel-Dominant)** | $(0.2, 0.6, 0.2)$ | 0.4827 | 0.9887 | 0.0092 | 1.0000 | 0.0183 | Candidate |
| **Set D (Drift-Balanced)** | $(0.3, 0.4, 0.3)$ | **0.6020** | **0.9923** | 0.0072 | 1.0000 | 0.0143 | **SELECTED (WINNER)** |

---

### E. Historical Policy Threshold Sweep ($T_3$ Test Set Evaluation)

Policies evaluated on $T_3$ using cutoffs frozen strictly from $T_1/T_2$:

| Policy Percentile | Policy Cutoff | Precision | Recall | F1 | False Positives | False Negatives | FPR | Policy Action |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **$p_{95}$ (Permissive)** | 0.8788 | 0.0500 | 0.5000 | 0.0909 | 95 | 5 | 0.04774 | Monitor / Soft Challenge |
| **$p_{97.5}$ (Balanced)** | 0.8938 | 0.0571 | 0.4000 | 0.1000 | 66 | 6 | 0.03317 | Step-Up Verification / Hold |
| **$p_{99}$ (Strict)** | 0.9028 | 0.0816 | 0.4000 | 0.1356 | 45 | 6 | 0.02261 | Block / Manual Review |
| **$p_{99.5}$ (Conservative)** | 0.9930 | 0.0000 | 0.0000 | 0.0000 | 0 | 10 | 0.00000 | Emergency Block |

---

## Four Scientifically Distinct System Capabilities

1. **Population Genome Drift**: *"What changed globally between T2 and T3?"* (Wasserstein distance & KS stat summary in `data/genome_drift_summary.json`).
2. **Account-Level Historical Drift**: *"Does this account differ from its historical genome baseline?"* (Standardized Z-scores in `genome_full.csv`).
3. **Relational Anomaly Detection**: *"Does this account have unusual infrastructure sharing relationships?"* (Graph projection co-usage score).
4. **Calibrated Risk Decision**: *"Given a frozen historical policy, should this account be investigated?"* (Calibrated hybrid risk layer & policy action).

---

## Setup & Usage

### 1. Install Dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run End-to-End Pipeline

To execute the complete pipeline, XAI engine, and unit tests:

```bash
python3 src/generate_data.py
python3 src/build_graph.py
python3 src/community.py
python3 src/features.py
python3 src/genome_drift.py
python3 src/model.py
python3 src/xai.py
```
