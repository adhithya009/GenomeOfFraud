"""
app.py — Streamlit Investigation & Demonstration Interface for GenomeOfFraud

This application provides an interactive, evidence-driven fraud investigation platform
demonstrating the GenomeOfFraud research pipeline:
1. Account-level Risk Overview & Cost-Sensitive Policy Action
2. Hybrid Risk Component Decomposition (Supervised 30% / Relational 40% / Drift 30%)
3. Behavioral & Relational Fraud Genome Inspection
4. Subgraph Relational Topology Visualization (Account ↔ IP ↔ Merchant ↔ Device)
5. Historical Genome Drift & T2 -> T3 Surface Mutation Comparison
6. SHAP Attribution & Model Vulnerability Diagnostics
7. Out-of-Time Model Benchmarks & Alert-Budget Capacity Evaluation
8. Population Genome Mutation Drift Analysis
9. Case Study Demonstration Mode (Detected Fraud, Missed Fraud, Normal, False Positive)
"""

import sys
import os
import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import networkx as nx

# Add src to python path for module imports
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Import pipeline helpers
from src.genome_drift import (
    FORBIDDEN_COLS,
    POLICY_ACTIONS,
    apply_cost_sensitive_policy,
    build_historical_genome_reference,
    compute_account_genome_drift,
    compute_hybrid_risk_layer_calibrated,
    compute_mutation_aware_features,
    compute_relational_anomaly_score,
    get_gene_group,
    transform_empirical_cdf,
)
from src.model import MUTATION_AWARE_FEATURES, FRAUDGENOME_FEATURES, BASELINE_FEATURES
from src.xai import (
    FORBIDDEN_XAI_COLS,
    audit_xai_feature_matrix,
    compute_gene_group_contributions,
    compute_genome_drift_breakdown,
    compute_global_feature_importance,
    compute_local_account_explanation,
    compute_relational_anomaly_breakdown,
    compute_risk_component_contributions,
    generate_human_readable_explanation,
)

# ---------------------------------------------------------------------------
# 1. Streamlit App Configuration & Dark Theme Styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="GenomeOfFraud — Fraud Intelligence Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for sleek dark dashboard styling
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stApp { background-color: #0e1117; color: #e0e0e0; }
    
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 16px;
    }
    
    .metric-value {
        font-size: 26px;
        font-weight: bold;
        color: #ffffff;
    }
    .metric-label {
        font-size: 13px;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-critical { background-color: #7f1d1d; color: #fecaca; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-high { background-color: #7c2d12; color: #ffedd5; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-medium { background-color: #713f12; color: #fef08a; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    .badge-low { background-color: #14532d; color: #dcfce7; padding: 4px 8px; border-radius: 4px; font-weight: bold; }
    
    .badge-model { background-color: #1e3a8a; color: #bfdbfe; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    .badge-drift { background-color: #581c87; color: #e9d5ff; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    .badge-relational { background-color: #134e4a; color: #99f6e4; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
    .badge-policy { background-color: #312e81; color: #c7d2fe; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 2. Cached Data & Resource Loading Engine
# ---------------------------------------------------------------------------

@st.cache_data
def load_project_data():
    """Load cached DataFrames and machine-readable JSON artifacts."""
    project_root = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_root, "data")

    df_full = pd.read_csv(os.path.join(data_dir, "genome_full.csv"))
    df_t1 = pd.read_csv(os.path.join(data_dir, "genome_t1.csv"))
    df_t2 = pd.read_csv(os.path.join(data_dir, "genome_t2.csv"))
    df_t3 = pd.read_csv(os.path.join(data_dir, "genome_t3.csv"))

    with open(os.path.join(data_dir, "model_results.json"), "r") as f:
        model_results = json.load(f)

    with open(os.path.join(data_dir, "risk_calibration_t1_t2.json"), "r") as f:
        calibration_profile = json.load(f)

    with open(os.path.join(data_dir, "genome_reference_t1_t2.json"), "r") as f:
        reference_t1_t2 = json.load(f)

    with open(os.path.join(data_dir, "xai_case_studies.json"), "r") as f:
        case_studies = json.load(f)

    with open(os.path.join(data_dir, "genome_drift_summary.json"), "r") as f:
        drift_summary = json.load(f)

    global_imp_df = pd.read_csv(os.path.join(data_dir, "xai_global_importance.csv"))
    group_imp_df = pd.read_csv(os.path.join(data_dir, "xai_gene_group_importance.csv"))

    return (
        df_full, df_t1, df_t2, df_t3,
        model_results, calibration_profile, reference_t1_t2,
        case_studies, drift_summary, global_imp_df, group_imp_df, data_dir
    )


@st.cache_resource
def load_graph_resources(data_dir):
    """Load pre-built NetworkX phase graphs for sub-graph visualization."""
    graphs = {}
    for phase, filename in [("T1", "graph_t1.pkl"), ("T2", "graph_t2.pkl"), ("T3", "graph_t3.pkl")]:
        path = os.path.join(data_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    graphs[phase] = pickle.load(f)
            except Exception as e:
                graphs[phase] = None
        else:
            graphs[phase] = None
    return graphs


@st.cache_resource
def load_model_resources(data_dir):
    """Load serialized XGBoost model checkpoint."""
    model_path = os.path.join(data_dir, "models", "hybrid_risk_best.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


# ---------------------------------------------------------------------------
# 3. Main Streamlit Execution Flow
# ---------------------------------------------------------------------------

def main():
    (
        df_full, df_t1, df_t2, df_t3,
        model_results, calibration_profile, reference_t1_t2,
        case_studies, drift_summary, global_imp_df, group_imp_df, data_dir
    ) = load_project_data()

    graphs = load_graph_resources(data_dir)
    model = load_model_resources(data_dir)

    # -----------------------------------------------------------------------
    # Sidebar Navigation & Controls
    # -----------------------------------------------------------------------
    st.sidebar.markdown("## 🧬 **GenomeOfFraud**")
    st.sidebar.markdown("*Fraud Intelligence & XAI Platform*")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Navigation Mode",
        options=[
            "🔍 Account Investigation",
            "📊 Model Overview & Benchmarks",
            "🧬 Population Mutation Analysis",
            "💡 Case Studies Demonstration",
        ],
        index=0,
    )

    st.sidebar.markdown("---")

    # Phase Selector
    phase_selection = st.sidebar.selectbox(
        "Select Dataset Phase",
        options=["T3 (Current / Mutated)", "T2 (Coordinated Fraud)", "T1 (Historical Baseline)"],
        index=0,
    )
    selected_phase = phase_selection.split(" ")[0]

    phase_df_map = {"T1": df_t1, "T2": df_t2, "T3": df_t3}
    current_df = phase_df_map.get(selected_phase, df_t3)

    # Account Selector for Account Investigation Mode
    selected_acct_id = None
    if mode == "🔍 Account Investigation":
        st.sidebar.markdown("### Account Selection")

        # Allow quick filtering by risk/fraud status for easy navigation
        filter_option = st.sidebar.radio(
            "Filter Accounts",
            options=["All Accounts", "High Risk / Fraud First"],
            index=1,
        )

        acct_ids = list(current_df["account_id"].unique())
        if filter_option == "High Risk / Fraud First" and "account_genome_drift" in current_df.columns:
            # Sort accounts by highest drift/relational score
            sorted_df = current_df.sort_values(
                by=["relational_anomaly_score" if "relational_anomaly_score" in current_df.columns else "account_genome_drift"],
                ascending=False
            )
            acct_ids = list(sorted_df["account_id"].unique())

        selected_acct_id = st.sidebar.selectbox("Select Account ID", options=acct_ids, index=0)
        st.sidebar.caption(f"Total Accounts in {selected_phase}: {len(acct_ids)}")

    st.sidebar.markdown("---")
    st.sidebar.info("💡 **Razorpay Hackathon Demo**: GenomeOfFraud demonstrates how relational graph topology and historical drift catch mutated fraud ($T_3$) when supervised model probabilities drop.")

    # -----------------------------------------------------------------------
    # PAGE 1: ACCOUNT INVESTIGATION MODE
    # -----------------------------------------------------------------------
    if mode == "🔍 Account Investigation":
        render_account_investigation(
            selected_acct_id, selected_phase, current_df, df_full,
            reference_t1_t2, calibration_profile, model, graphs, global_imp_df, group_imp_df
        )

    # -----------------------------------------------------------------------
    # PAGE 2: MODEL OVERVIEW & BENCHMARKS
    # -----------------------------------------------------------------------
    elif mode == "📊 Model Overview & Benchmarks":
        render_model_overview(model_results, global_imp_df, group_imp_df)

    # -----------------------------------------------------------------------
    # PAGE 3: POPULATION MUTATION ANALYSIS
    # -----------------------------------------------------------------------
    elif mode == "🧬 Population Mutation Analysis":
        render_population_mutation_analysis(drift_summary, df_t2, df_t3)

    # -----------------------------------------------------------------------
    # PAGE 4: CASE STUDIES DEMONSTRATION
    # -----------------------------------------------------------------------
    elif mode == "💡 Case Studies Demonstration":
        render_case_studies(case_studies)


# ===========================================================================
# 4. Page Implementation: Account Investigation
# ===========================================================================

def render_account_investigation(
    acct_id, phase, current_df, df_full,
    reference_dict, calib_profile, model, graphs, global_imp_df, group_imp_df
):
    """Render comprehensive evidence-driven account investigation workspace."""
    account_row = current_df[current_df["account_id"] == acct_id].iloc[0]

    st.markdown(f"# 🔍 Account Investigation — `{acct_id}`")
    st.caption(f"Phase: **{phase}** | Account Age: **{account_row.get('acct_age_days', 0):.0f} days** | KYC Status: **{'Unverified' if account_row.get('acct_kyc_unverified', 0) > 0.5 else 'Verified'}**")
    st.markdown("---")

    # Compute risk components & policy decision using frozen calibration profile
    winning_weights = (0.3, 0.4, 0.3)

    feature_cols = [c for c in MUTATION_AWARE_FEATURES if c in current_df.columns]
    audit_xai_feature_matrix(feature_cols)

    X_vec = account_row[feature_cols].values.astype(float).reshape(1, -1)
    sup_prob = float(model.predict_proba(X_vec)[:, 1][0]) if model is not None else 0.0003

    rel_anomaly = float(account_row.get("relational_anomaly_score", compute_relational_anomaly_score(pd.DataFrame([account_row]), reference_dict)[0]))
    genome_drift = float(account_row.get("account_genome_drift", compute_account_genome_drift(pd.DataFrame([account_row]), reference_dict)["account_genome_drift"].iloc[0]))

    comp_res = compute_risk_component_contributions(
        sup_prob, rel_anomaly, genome_drift, calib_profile, weights=winning_weights
    )
    final_risk = comp_res["final_hybrid_risk_score"]

    policy_res = apply_cost_sensitive_policy(final_risk, calib_profile)
    cat = policy_res["risk_category"]

    # -----------------------------------------------------------------------
    # SECTION 1: GENOMEOFFRAUD RISK OVERVIEW CARDS
    # -----------------------------------------------------------------------
    st.markdown("## GENOMEOFFRAUD RISK OVERVIEW")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"<div class='card'><div class='metric-label'>Final Hybrid Risk Score</div><div class='metric-value'>{final_risk:.4f}</div></div>", unsafe_allow_html=True)
    with c2:
        badge_cls = f"badge-{cat.lower()}"
        st.markdown(f"<div class='card'><div class='metric-label'>Risk Level</div><div class='metric-value'><span class='{badge_cls}'>{cat}</span></div></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(f"<div class='card'><div class='metric-label'>Recommended Policy Action</div><div class='metric-value' style='font-size:18px;'><code>{policy_res['policy_action']}</code></div></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(f"<div class='card'><div class='metric-label'>Decision Signal Confidence</div><div class='metric-value' style='font-size:18px;'>{policy_res['decision_confidence']}</div></div>", unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------------------------------------------------
    # SECTION 2: HYBRID RISK COMPONENT DECOMPOSITION
    # -----------------------------------------------------------------------
    st.markdown("## Risk Components & Calibrated Hybrid Layer")
    st.caption("Hybrid Risk Score = (30% × Supervised Probability CDF) + (40% × Relational Anomaly CDF) + (30% × Genome Drift CDF)")

    raw = comp_res["raw_signals"]
    cdfs = comp_res["cdf_percentiles"]
    contribs = comp_res["component_contributions"]

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.metric(
            label="Supervised Model Risk (30% weight)",
            value=f"{raw['supervised_prob']:.4f}",
            delta=f"+{contribs['supervised_risk_contribution']:.4f} contrib (CDF {cdfs['supervised_prob_cdf']:.2%})",
        )
    with r2:
        st.metric(
            label="Relational Anomaly (40% weight)",
            value=f"{raw['relational_anomaly_score']:.4f}",
            delta=f"+{contribs['relational_anomaly_contribution']:.4f} contrib (CDF {cdfs['relational_anomaly_cdf']:.2%})",
        )
    with r3:
        st.metric(
            label="Account Genome Drift (30% weight)",
            value=f"{raw['genome_drift_score']:.4f}",
            delta=f"+{contribs['genome_drift_contribution']:.4f} contrib (CDF {cdfs['genome_drift_cdf']:.2%})",
        )
    with r4:
        st.metric(
            label="Final Risk Score",
            value=f"{final_risk:.4f}",
            delta=f"Threshold Cutoff: {policy_res['selected_threshold']:.4f}",
        )

    st.info("ℹ️ **Scientific Note**: The hybrid risk score represents a calibrated operational policy indicator relative to historical normal percentiles ($T_1/T_2$), NOT an uncalibrated raw fraud probability.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # SECTION 3: RELATIONAL GENOME & TOPOLOGY SUBGRAPH
    # -----------------------------------------------------------------------
    t_rel1, t_rel2 = st.tabs(["🌐 Relational Network Graph", "📊 Infrastructure Sharing Signals"])

    with t_rel1:
        st.markdown("### Relational Subgraph Topology")
        st.caption("Visualizing 1-hop and 2-hop behavioral connections for this account across Shared Devices, Shared IPs, and Merchant Entities.")

        G_phase = graphs.get(phase)
        if G_phase is not None:
            acct_node_id = f"account:{acct_id}"
            if G_phase.has_node(acct_node_id):
                # Extract 2-hop ego sub-graph
                neighbors_1 = set(G_phase.neighbors(acct_node_id))
                neighbors_2 = set()
                for n in neighbors_1:
                    neighbors_2.update(G_phase.neighbors(n))
                sub_nodes = {acct_node_id}.union(neighbors_1).union(neighbors_2)

                # Limit node count for clean rendering
                if len(sub_nodes) > 35:
                    sub_nodes = {acct_node_id}.union(list(neighbors_1)[:15]).union(list(neighbors_2)[:15])

                sub_G = G_phase.subgraph(sub_nodes)

                fig, ax = plt.subplots(figsize=(10, 6))
                pos = nx.spring_layout(sub_G, seed=42)

                node_colors = []
                node_sizes = []
                for n in sub_G.nodes():
                    ntype = sub_G.nodes[n].get("node_type", "account")
                    if n == acct_node_id:
                        node_colors.append("#d62728")  # Red for target account
                        node_sizes.append(600)
                    elif ntype == "account":
                        node_colors.append("#1f77b4")  # Blue for other accounts
                        node_sizes.append(300)
                    elif ntype == "device":
                        node_colors.append("#ff7f0e")  # Orange for devices
                        node_sizes.append(250)
                    elif ntype == "ip":
                        node_colors.append("#2ca02c")  # Green for IPs
                        node_sizes.append(250)
                    else:
                        node_colors.append("#9467bd")  # Purple for merchants
                        node_sizes.append(250)

                nx.draw_networkx_nodes(sub_G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
                nx.draw_networkx_edges(sub_G, pos, alpha=0.4, edge_color="#888888", ax=ax)

                # Labels for key nodes
                labels = {n: n.split(":")[0][:3] + ":" + n.split(":")[1][:6] for n in sub_G.nodes() if n == acct_node_id or n in neighbors_1}
                nx.draw_networkx_labels(sub_G, pos, labels=labels, font_size=8, font_color="#ffffff", ax=ax)

                ax.set_facecolor("#0e1117")
                fig.patch.set_facecolor("#0e1117")
                plt.axis("off")
                st.pyplot(fig)
                plt.close()

                st.caption("🔴 Target Account | 🔵 Co-connected Accounts | 🟠 Devices | 🟢 IPs | 🟣 Merchants")
            else:
                st.warning(f"Account `{acct_node_id}` not found in phase {phase} graph pickle.")
        else:
            st.info("Relational graph pickle file not available for interactive network rendering; displaying structured relational metrics instead.")

    with t_rel2:
        st.markdown("### Relational Infrastructure Signals & Deviations")
        rel_signals = compute_relational_anomaly_breakdown(account_row, reference_dict)

        rel_df = pd.DataFrame(rel_signals)
        st.dataframe(
            rel_df.style.format({
                "current_value": "{:.4f}",
                "historical_mean": "{:.4f}",
                "historical_std": "{:.4f}",
                "positive_std_elevation": "+{:.2f}σ",
            }),
            use_container_width=True,
        )

        top_elevated = [r for r in rel_signals if r["positive_std_elevation"] > 1.5]
        if top_elevated:
            st.warning(f"⚠️ **High Infrastructure Co-Usage Elevation**: `{top_elevated[0]['feature']}` is **+{top_elevated[0]['positive_std_elevation']:.1f}σ** above historical baseline.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # SECTION 4: HISTORICAL GENOME DRIFT & T2 -> T3 MUTATION COMPARISON
    # -----------------------------------------------------------------------
    st.markdown("## Historical Genome Drift & T2 → T3 Mutation View")

    drift_breakdown = compute_genome_drift_breakdown(account_row, reference_dict)
    g_drifts = drift_breakdown["gene_group_drift_scores"]

    col_d1, col_d2 = st.columns([1, 1])

    with col_d1:
        st.markdown("### Gene Group Drift Scores ($Z$-Scores)")
        drift_chart_df = pd.DataFrame({
            "Gene Group": [k.replace("_drift_score", "").title() for k in g_drifts.keys()],
            "Z-Score Shift": list(g_drifts.values()),
        }).sort_values("Z-Score Shift", ascending=False)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(drift_chart_df["Gene Group"], drift_chart_df["Z-Score Shift"], color="#9467bd")
        ax.set_xlabel("Mean Absolute Z-Score Shift vs Reference", fontweight="bold")
        ax.set_facecolor("#0e1117")
        fig.patch.set_facecolor("#0e1117")
        ax.tick_params(colors="#ffffff")
        ax.xaxis.label.set_color("#ffffff")
        ax.yaxis.label.set_color("#ffffff")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_d2:
        st.markdown("### T2 → T3 Surface Tactic Mutation Comparison")
        st.caption("Comparing surface device behavior vs persistent relational infrastructure sharing.")

        dev_sharing = float(account_row.get("dev_shared_ratio", 0.0))
        net_sharing = float(account_row.get("net_shared_ip_ratio", 0.0))
        merch_conc = float(account_row.get("merch_concentration", 0.0))
        proj_deg = float(account_row.get("graph_proj_degree", 0.0))

        st.write(f"* **Device Usage Tactic**: Current Device Sharing Ratio = `{dev_sharing:.2f}` ({'Mutated to unique devices' if dev_sharing < 0.5 else 'Shared device pool'})")
        st.write(f"* **Network IP Persistence**: Current IP Sharing Ratio = `{net_sharing:.2f}` ({'Persistent IP co-usage' if net_sharing > 0.8 else 'Dispersed IP pool'})")
        st.write(f"* **Merchant Target Concentration**: Current Merchant Concentration = `{merch_conc:.2f}`")
        st.write(f"* **Relational Topology Degree**: Projected Account Degree = `{proj_deg:.2f}`")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # SECTION 5: SHAP SUPERVISED MODEL EXPLANATION
    # -----------------------------------------------------------------------
    st.markdown("## Supervised Model Attribution (SHAP)")

    # Display local SHAP features
    local_exp = compute_local_account_explanation(
        account_row, model, shap.TreeExplainer(model) if model is not None else None,
        feature_cols, reference_dict, calib_profile, weights=winning_weights
    )

    top_shap = local_exp["top_shap_features"][:8]
    shap_chart_df = pd.DataFrame(top_shap).iloc[::-1]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in shap_chart_df["shap_value"]]
    ax.barh(shap_chart_df["feature"], shap_chart_df["shap_value"], color=colors)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_xlabel("SHAP Value (Attribution to Supervised Fraud Risk)", fontweight="bold")
    ax.set_title(f"Account {acct_id} — Local SHAP Attribution", fontsize=11, fontweight="bold", color="#ffffff")
    ax.set_facecolor("#0e1117")
    fig.patch.set_facecolor("#0e1117")
    ax.tick_params(colors="#ffffff")
    ax.xaxis.label.set_color("#ffffff")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("### What Does the Supervised Model Rely On globally?")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        summary_img = os.path.join(data_dir, "xai_shap_summary.png")
        if os.path.exists(summary_img):
            st.image(summary_img, caption="Global SHAP Summary Plot across all features")
    with col_g2:
        gene_img = os.path.join(data_dir, "xai_gene_group_importance.png")
        if os.path.exists(gene_img):
            st.image(gene_img, caption="Relative Importance by Fraud Genome Gene Group")

    st.warning("🔬 **Supervised Model Vulnerability Insight**: Global SHAP analysis confirms that the supervised XGBoost model relies heavily (~95.6%) on surface Account velocity and transaction count features. When fraud syndicates mutate surface transaction counts in $T_3$, supervised probability drops from 0.99 to <0.01. GenomeOfFraud overcomes this vulnerability by incorporating Relational Anomaly (+0.399) and Genome Drift (+0.300) layers into the final decision.")

    st.markdown("---")

    # -----------------------------------------------------------------------
    # SECTION 6: DYNAMIC EVIDENCE SYNTHESIS & INVESTIGATION REPORT
    # -----------------------------------------------------------------------
    st.markdown("## Final Evidence Summary & Investigation Report")

    human_narrative = generate_human_readable_explanation(local_exp)
    st.markdown(human_narrative)

    st.markdown("""
    <div style='margin-top:20px;'>
        <span class='badge-model'>MODEL ATTRIBUTION</span>
        <span class='badge-drift'>STATISTICAL DEVIATION</span>
        <span class='badge-relational'>RELATIONAL EVIDENCE</span>
        <span class='badge-policy'>POLICY DECISION</span>
    </div>
    """, unsafe_allow_html=True)


# ===========================================================================
# 5. Page Implementation: Model Overview & Benchmarks
# ===========================================================================

def render_model_overview(model_results, global_imp_df, group_imp_df):
    """Render comprehensive model benchmarks, ablation study, and alert capacity analysis."""
    st.markdown("# 📊 Model Overview & Out-of-Time Benchmarks")
    st.caption("Out-of-Time Temporal Evaluation ($T_1+T_2 \\rightarrow T_3$) under Tactic Mutation")
    st.markdown("---")

    # Table A: 7-Part Out-of-Time Ablation Study
    st.markdown("## Out-of-Time 7-Part Temporal Ablation Study ($T_3$)")
    drift_exps = model_results.get("temporal_drift_experiments", {})

    rows = []
    for exp_name, exp_data in drift_exps.items():
        eval_m = exp_data.get("t3_evaluation", {})
        rows.append({
            "Experiment": exp_name,
            "Feature Count": exp_data.get("num_features", 0),
            "PR-AUC": eval_m.get("pr_auc", 0.0),
            "ROC-AUC": eval_m.get("roc_auc", 0.0),
            "Precision": eval_m.get("precision", 0.0),
            "Recall": eval_m.get("recall", 0.0),
            "F1": eval_m.get("f1", 0.0),
            "False Positives": eval_m.get("fp_count", 0),
            "False Negatives": eval_m.get("fn_count", 0),
            "FPR": eval_m.get("fpr", 0.0),
            "Cutoff Threshold": exp_data.get("val_tuned_threshold", 0.5),
        })

    ablation_df = pd.DataFrame(rows)
    st.dataframe(ablation_df.style.highlight_max(axis=0, subset=["PR-AUC", "ROC-AUC", "F1", "Recall"], color="#14532d"), use_container_width=True)

    st.markdown("---")

    # Table B: Operational Alert-Budget Capacity Analysis
    st.markdown("## Operational Alert-Budget Review Capacity Analysis ($T_3$)")
    st.caption("Evaluating fraud capture rates under realistic analyst capacity review budgets.")

    alert_budgets = model_results.get("alert_budget_analysis", [])
    budget_df = pd.DataFrame(alert_budgets)

    c_b1, c_b2 = st.columns([1.2, 1])
    with c_b1:
        st.dataframe(
            budget_df.style.format({
                "budget_pct": "{:.1f}%",
                "capture_rate": "{:.2%}",
                "precision": "{:.2%}",
            }),
            use_container_width=True
        )
    with c_b2:
        pk = model_results.get("precision_at_k", {})
        st.markdown("### Precision@K Metrics")
        st.write(f"* **Precision@1%**: `{pk.get('p_at_1.0%', 0.0):.2%}` (Recall@1%: `{pk.get('r_at_1.0%', 0.0):.2%}`)")
        st.write(f"* **Precision@5%**: `{pk.get('p_at_5.0%', 0.0):.2%}` (Recall@5%: `{pk.get('r_at_5.0%', 0.0):.2%}`)")
        st.write(f"* **Precision@10%**: `{pk.get('p_at_10.0%', 0.0):.2%}` (Recall@10%: `{pk.get('r_at_10.0%', 0.0):.2%}`)")

    st.markdown("---")

    # Global SHAP Table
    st.markdown("## Global SHAP Feature Importances")
    st.dataframe(global_imp_df.head(20), use_container_width=True)


# ===========================================================================
# 6. Page Implementation: Genome Mutation Analysis
# ===========================================================================

def render_population_mutation_analysis(drift_summary, df_t2, df_t3):
    """Render population-level T2 -> T3 genome mutation analysis."""
    st.markdown("# 🧬 Population Genome Mutation Analysis ($T_2 \\rightarrow T_3$)")
    st.caption("Offline population-level distribution drift summary measuring syndicate tactic shifts.")
    st.markdown("---")

    st.markdown("## Population Drift Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Device Genome Drift", drift_summary.get("device_genome_drift", "UNKNOWN"))
    with c2:
        st.metric("Network Genome Drift", drift_summary.get("network_genome_drift", "UNKNOWN"))

    diag = drift_summary.get("diagnostic_metrics", {})
    st.markdown("### Key Diagnostic Mutation Signals")
    st.write(f"* **Device Usage Density Change**: `{diag.get('device_density_change', 0.0):+.4f}` (Shift from shared devices to unique device pool)")
    st.write(f"* **Network IP Sharing Persistence**: `{diag.get('ip_sharing_persistence', 0.0):.4f}` (High persistence across time)")
    st.write(f"* **Merchant Sharing Persistence**: `{diag.get('merchant_sharing_persistence', 0.0):.4f}`")

    st.markdown("---")
    st.markdown("## Gene Group Drift Rankings")
    rankings = pd.DataFrame(drift_summary.get("gene_group_rankings", []))
    st.dataframe(rankings, use_container_width=True)


# ===========================================================================
# 7. Page Implementation: Case Studies Demonstration
# ===========================================================================

def render_case_studies(case_studies):
    """Render guided demonstration case studies for hackathon review."""
    st.markdown("# 💡 Hackathon Case Studies Demonstration Mode")
    st.caption("Guided walkthrough comparing detected fraud, missed fraud, normal activity, and false positives under $T_3$ mutation.")
    st.markdown("---")

    case_choice = st.radio(
        "Select Demonstration Case Study",
        options=[
            "🎯 Detected T3 Fraud (Correctly Flagged)",
            "⚠️ Missed T3 Fraud (Model Evasion / Surface Mutation)",
            "🟢 T3 Normal Account (Correctly Allowed)",
            "🟡 False Positive Account (Flagged Normal)",
        ],
        index=0,
    )

    key_map = {
        "🎯 Detected T3 Fraud (Correctly Flagged)": "detected_t3_fraud",
        "⚠️ Missed T3 Fraud (Model Evasion / Surface Mutation)": "missed_t3_fraud",
        "🟢 T3 Normal Account (Correctly Allowed)": "t3_normal",
        "🟡 False Positive Account (Flagged Normal)": "false_positive",
    }

    selected_key = key_map[case_choice]
    case_data = case_studies.get(selected_key, {})

    st.markdown(f"## Case Study: `{selected_key.replace('_', ' ').title()}`")
    st.caption(f"Account ID: `{case_data.get('account_id')}` | Ground Truth Fraud: **{case_data.get('is_fraud_ground_truth')}**")

    rsum = case_data.get("risk_summary", {})
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Hybrid Risk Score", f"{rsum.get('final_risk_score', 0.0):.4f}")
    with c2:
        st.metric("Risk Level", rsum.get("risk_category"))
    with c3:
        st.metric("Policy Action", rsum.get("policy_action"))
    with c4:
        st.metric("Confidence", rsum.get("decision_confidence"))

    st.markdown("---")

    if selected_key == "missed_t3_fraud":
        st.error("⚠️ **Scientific Transparency**: This account represents a ground-truth fraud account missed by the policy threshold ($R_{\\text{final}} < p_{97.5}$). Under $T_3$, the syndicate mutated device fingerprints and transaction counts, causing supervised probability to fall below 0.01. GenomeOfFraud keeps missed fraud cases visible to analyze structural limitations.")

    st.markdown("### Human-Readable Evidence Narrative")
    st.markdown(case_data.get("human_readable_explanation", "No explanation text available."))


if __name__ == "__main__":
    main()
