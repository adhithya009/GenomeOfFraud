# GenomeOfFraud — Hackathon Demo Script & Walkthrough (3–5 Minutes)

This document provides a step-by-step presentation script for demonstrating the **GenomeOfFraud** platform to hackathon judges and researchers.

---

## Presentation Goals

1. **Demonstrate Tactic Evasion**: Show how traditional supervised ML models fail when fraud syndicates mutate surface transaction features ($T_3$).
2. **Demonstrate Relational & Drift Intelligence**: Show how GenomeOfFraud's Relational Anomaly (+3.2$\sigma$ projection degree elevation) and Account Genome Drift layers catch mutated fraud.
3. **Demonstrate Explainable AI**: Walk through the end-to-end evidence taxonomy linking risk score, policy action, SHAP attributions, network graph, and human-readable evidence summaries.
4. **Demonstrate Scientific Honesty**: Highlight the missed-fraud case study to show model boundary conditions.

---

## 3–5 Minute Demonstration Flow

### Step 1: Open the Dashboard & Select a Mutated $T_3$ Fraud Account (30 Seconds)
- Launch the app: `streamlit run app.py`
- Set Phase to **`T3 (Current / Mutated)`**.
- Select Account ID: `42a8a15c-0c00-4fb6-9df7-e010074b21cd`.
- **Script**: *"Under Phase T3, fraud syndicates intentionally mutate their surface transaction behavior — switching from shared devices to unique device fingerprints and lowering transaction velocity to evade traditional rule engines and supervised models."*

### Step 2: Show Risk Overview & Policy Action (30 Seconds)
- Point to the top metric cards:
  - **Final Hybrid Risk Score**: `0.9958`
  - **Risk Level**: `CRITICAL` (Red Badge)
  - **Recommended Policy Action**: `BLOCK_MANUAL_REVIEW`
  - **Decision Confidence**: `HIGH`
- **Script**: *"Despite the surface mutation, GenomeOfFraud assigns this account a CRITICAL risk level and triggers an automatic BLOCK policy action."*

### Step 3: Explain the Hybrid Risk Component Decomposition (45 Seconds)
- Scroll to the **Risk Components** section:
  - Supervised Risk (30% weight): Raw score `0.000282` (Contribution: `+0.2967`)
  - Relational Anomaly (40% weight): Raw score `1.500782` (Contribution: `+0.3991`)
  - Genome Drift (30% weight): Raw score `2.472168` (Contribution: `+0.3000`)
- **Script**: *"Notice the supervised model risk probability dropped to 0.000282 because the syndicate mutated device count and velocity. If we relied solely on supervised ML, this account would be allowed! However, our Relational Anomaly score is 1.5007 (+0.3991 contribution) and Genome Drift is 2.4721 (+0.3000 contribution), bringing the final hybrid score to 0.9958."*

### Step 4: Show the Relational Subgraph Topology (45 Seconds)
- Click on the **`🌐 Relational Network Graph`** tab.
- Point to the red target account node connected to green IP nodes, purple Merchant nodes, and blue co-connected accounts.
- Click on the **`📊 Infrastructure Sharing Signals`** tab to show `graph_proj_degree` is **+3.22σ** above historical baseline.
- **Script**: *"Here is the key insight of GenomeOfFraud: while surface device usage mutated, the underlying relational infrastructure topology remains suspicious. The account has a +3.22σ elevation in projected account-account graph connectivity and co-uses infrastructure with 4 other accounts."*

### Step 5: Show SHAP & Global Model Vulnerability (45 Seconds)
- Scroll to the **Supervised Model Attribution (SHAP)** section.
- Point to the global SHAP plot and scientific callout box.
- **Script**: *"Our global SHAP analysis proves that supervised tree models rely 95.6% on surface account velocity features. That's why supervised probability fails under T3 mutation. GenomeOfFraud explicitly decouples surface behavior from relational structure."*

### Step 6: Case Studies Demonstration Mode & Scientific Honesty (45 Seconds)
- Switch Navigation Mode in the sidebar to **`💡 Case Studies Demonstration`**.
- Select **`⚠️ Missed T3 Fraud (Model Evasion / Surface Mutation)`**.
- Point to the missed fraud account (`a06c2b65-08c2-4d1c-b208-e2c1f7f0ce5f`) with Risk Score `0.8524` (`ALLOW`).
- **Script**: *"We do not hide model failures. Here is a missed T3 fraud account where the syndicate mutated both device usage and IP pool sufficiently to stay just below our p95 policy threshold. Exposing missed fraud provides critical diagnostic transparency for security teams."*

---

## Key Takeaway Summary for Judges

| Dimension | Traditional Supervised Model | GenomeOfFraud Platform |
| :--- | :--- | :--- |
| **Reaction to Surface Mutation ($T_3$)** | Fails (Probability drops to <0.01) | **Succeeds (Hybrid risk score 0.9958)** |
| **Relational Infrastructure Tracking** | Ignored / Overfitted | **Explicit graph topology & $+Z\sigma$ anomaly elevation** |
| **Explainability** | Generic SHAP plot | **Multilayer evidence taxonomy (SHAP + Drift + Relational + Policy)** |
| **Calibration** | Uncalibrated probabilities | **Frozen historical ECDF percentiles ($T_1/T_2$)** |
