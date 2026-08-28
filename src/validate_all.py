"""
src/validate_all.py — Unified Master Automated Test Runner for GenomeOfFraud

This master validation runner executes all 18 unit, integration, causality, leakage,
and integrity test suites across the entire GenomeOfFraud research pipeline.

Execution Summary:
- Suite 1: Dataset Integrity & Behavioral Topology Verification (4 tests)
- Suite 2: Zero Target / Metadata Leakage Verification (2 tests)
- Suite 3: Temporal Immutability & Scoring Determinism Verification (2 tests)
- Suite 4: Explainable AI (XAI) Guardrails & Accounting Verification (7 tests)
- Suite 5: Streamlit Interface & Data Reconciliation Verification (3 tests)
"""

import os
import sys

# Ensure src directory is in sys.path
src_dir = os.path.dirname(os.path.abspath(__file__))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from genome_drift import (
    run_scoring_stability_unit_tests,
    run_temporal_immutability_unit_tests,
)
from model import (
    FRAUDGENOME_FEATURES,
    MUTATION_AWARE_FEATURES,
    load_features,
    audit_features,
    train_xgboost,
    prepare_temporal_split,
)
from xai import (
    audit_xai_feature_matrix,
    generate_shap_explanations,
    compute_local_account_explanation,
    run_xai_validation_tests,
)
from validate_streamlit import run_streamlit_validation_tests


def run_all_master_validation_tests():
    print("\n" + "=" * 80)
    print("  GENOMEOFFRAUD — UNIFIED MASTER AUTOMATED TEST SUITE (18 TESTS TOTAL)")
    print("=" * 80 + "\n")

    # Suite 1 & 2: Dataset Integrity & Feature Audit
    print("  [SUITE 1 & 2] Running Dataset & Zero-Leakage Feature Audit...")
    df, data_dir = load_features()
    audit_features(df)
    feature_cols = [c for c in MUTATION_AWARE_FEATURES if c in df.columns]
    audit_xai_feature_matrix(feature_cols)

    # Suite 3: Temporal Immutability & Scoring Stability Tests
    print("  [SUITE 3] Running Temporal Immutability & Scoring Stability Tests...")
    run_temporal_immutability_unit_tests()
    run_scoring_stability_unit_tests()

    # Suite 4: XAI Validation Tests
    print("  [SUITE 4] Running Explainable AI (XAI) Test Suite...")
    train_df, test_df = prepare_temporal_split(df)
    X_train = train_df[feature_cols].values
    y_train = train_df["is_fraud_account"].astype(int).values
    model, _ = train_xgboost(X_train, y_train, seed=42)

    from genome_drift import build_historical_genome_reference, build_historical_calibration_profile, compute_account_genome_drift, compute_relational_anomaly_score
    ref_t1_t2 = build_historical_genome_reference(train_df, reference_name="t1_t2_master")
    hist_norm = train_df[~train_df["is_fraud_account"].astype(bool)].reset_index(drop=True)
    norm_prob = model.predict_proba(hist_norm[feature_cols].values)[:, 1]
    norm_rel = compute_relational_anomaly_score(hist_norm, ref_t1_t2)
    norm_drift = hist_norm["account_genome_drift"].values if "account_genome_drift" in hist_norm.columns else np.zeros(len(hist_norm))
    calib = build_historical_calibration_profile(hist_norm, norm_prob, norm_rel, norm_drift)

    import shap
    explainer = shap.TreeExplainer(model)
    run_xai_validation_tests(test_df, feature_cols, model, explainer, ref_t1_t2, calib)

    # Suite 5: Streamlit Interface & Data Reconciliation Tests
    print("  [SUITE 5] Running Streamlit Interface & Data Reconciliation Test Suite...")
    run_streamlit_validation_tests()

    print("=" * 80)
    print("  🎉 ALL 18 UNIFIED MASTER VALIDATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_all_master_validation_tests()
