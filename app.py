"""Interactive ProtocolLens demonstration dashboard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protocollens.eligibility import criteria_as_dicts, eligibility_features  # noqa: E402
from protocollens.modelling import explain_linear_prediction  # noqa: E402
from protocollens.outcomes import categorize_termination_reason  # noqa: E402
from protocollens.retrieval import (  # noqa: E402
    SemanticTrialRetriever,
    TfidfTrialRetriever,
    summarize_judgments,
)
from protocollens.review import (  # noqa: E402
    apply_pair_judgment,
    review_progress,
    save_review_progress,
)

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "trials_with_features.csv"
RETRIEVAL_DIR = PROJECT_ROOT / "artifacts" / "retrieval"
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
REVIEW_PATH = PROJECT_ROOT / "data" / "processed" / "retrieval_evaluation.csv"

st.set_page_config(
    page_title="ProtocolLens",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px;}
    .context-box {background:#f0f7ff; border-left:4px solid #2878d0;
        color:#16324f; padding:0.9rem 1rem; border-radius:0.4rem; margin-bottom:1rem;}
    .warning-box {background:#fff8e6; border-left:4px solid #d99a00;
        color:#4a3600; padding:0.9rem 1rem; border-radius:0.4rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_trials() -> pd.DataFrame:
    """Load the processed cohort once and reuse it across dashboard reruns."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_retrievers():
    """Load the lexical and semantic indexes once for the Streamlit process."""
    return (
        TfidfTrialRetriever.load(RETRIEVAL_DIR / "tfidf.joblib"),
        SemanticTrialRetriever.load(RETRIEVAL_DIR / "semantic"),
    )


@st.cache_data
def load_model_outputs():
    comparison = pd.read_csv(MODEL_DIR / "model_comparison.csv")
    reliability = pd.read_csv(MODEL_DIR / "reliability.csv")
    importance = pd.read_csv(MODEL_DIR / "global_feature_importance.csv")
    subgroup = pd.read_csv(MODEL_DIR / "subgroup_metrics.csv")
    validation_comparison = json.loads(
        (MODEL_DIR / "validation_comparison.json").read_text(encoding="utf-8")
    )
    summary = json.loads((MODEL_DIR / "training_summary.json").read_text(encoding="utf-8"))
    text_leakage_audit = json.loads(
        (MODEL_DIR / "text_leakage_audit.json").read_text(encoding="utf-8")
    )
    return (
        comparison,
        reliability,
        importance,
        subgroup,
        validation_comparison,
        summary,
        text_leakage_audit,
    )


@st.cache_resource
def load_best_uncalibrated_model():
    """Load the validation-selected Logistic Regression pipeline once."""
    return joblib.load(MODEL_DIR / "best_uncalibrated.joblib")


def header(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def display_text(value: object) -> str:
    """Return readable review context for blank registry fields."""
    if pd.isna(value):
        return "Not reported"
    text = str(value).strip()
    return text or "Not reported"


def overview_page(frame: pd.DataFrame) -> None:
    header(
        "ProtocolLens",
        "Interpretable eligibility intelligence for comparable-trial retrieval and retrospective status modelling",
    )
    st.markdown(
        """
        <div class="context-box"><b>Core question:</b> Does information in
        eligibility text add useful signal beyond ordinary structured registry
        fields when distinguishing completed from terminated studies?</div>
        """,
        unsafe_allow_html=True,
    )
    counts = frame["overall_status"].value_counts()
    columns = st.columns(4)
    columns[0].metric("Trials", f"{len(frame):,}", border=True)
    columns[1].metric("Completed", f"{counts.get('COMPLETED', 0):,}", border=True)
    columns[2].metric("Terminated", f"{counts.get('TERMINATED', 0):,}", border=True)
    columns[3].metric("Therapeutic areas", frame["query_conditions"].nunique(), border=True)

    left, right = st.columns(2)
    with left:
        area_counts = (
            frame.groupby(["query_conditions", "overall_status"]).size().reset_index(name="trials")
        )
        fig = px.bar(
            area_counts,
            x="query_conditions",
            y="trials",
            color="overall_status",
            barmode="group",
            title="Cohort composition",
            labels={"query_conditions": "Therapeutic area", "overall_status": "Status"},
            color_discrete_map={"COMPLETED": "#2878d0", "TERMINATED": "#e26a45"},
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.histogram(
            frame,
            x="complexity_index",
            color="overall_status",
            nbins=25,
            barmode="overlay",
            opacity=0.65,
            title="Eligibility complexity profile",
            labels={"complexity_index": "Descriptive complexity index"},
            color_discrete_map={"COMPLETED": "#2878d0", "TERMINATED": "#e26a45"},
        )
        st.plotly_chart(fig, width="stretch")

    terminated = frame[frame["overall_status"] == "TERMINATED"].copy()
    terminated["reason_category"] = terminated["why_stopped"].map(categorize_termination_reason)
    reason_counts = terminated["reason_category"].value_counts().reset_index()
    reason_counts.columns = ["reason_category", "trials"]
    fig = px.bar(
        reason_counts.sort_values("trials"),
        x="trials",
        y="reason_category",
        orientation="h",
        title="Reported reasons for termination (descriptive only)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Reason stopped is recorded after the outcome and is excluded from every model feature pipeline."
    )

    st.markdown(
        """
        <div class="warning-box"><b>Interpretation boundary:</b> registry status
        is the experimental label. This is not a clinical-success model, causal
        analysis, or medical decision system.</div>
        """,
        unsafe_allow_html=True,
    )


def eligibility_page(frame: pd.DataFrame) -> None:
    header(
        "Eligibility analyzer",
        "Turn unstructured inclusion/exclusion text into an auditable complexity profile",
    )
    default_text = str(frame.iloc[0]["eligibility_criteria"])
    text = st.text_area("Eligibility criteria", value=default_text, height=320)
    features = eligibility_features(text)
    criteria = criteria_as_dicts(text)
    columns = st.columns(5)
    columns[0].metric("Criteria", features["criterion_count"], border=True)
    columns[1].metric("Inclusion", features["inclusion_count"], border=True)
    columns[2].metric("Exclusion", features["exclusion_count"], border=True)
    columns[3].metric("Words", features["eligibility_word_count"], border=True)
    columns[4].metric(
        "Complexity", f"{features['complexity_index']:.1f}/100", border=True
    )

    feature_frame = pd.DataFrame(
        [
            {"component": key.replace("_", " ").title(), "value": value}
            for key, value in features.items()
            if key.endswith("_count") and key not in {"eligibility_character_count"}
        ]
    )
    left, right = st.columns([1, 1.35])
    with left:
        fig = px.bar(
            feature_frame,
            x="value",
            y="component",
            orientation="h",
            title="Transparent feature counts",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        st.subheader("Parsed criteria")
        if criteria:
            st.dataframe(pd.DataFrame(criteria), width="stretch", hide_index=True)
        else:
            st.info("Enter eligibility text to parse it.")
    st.caption(
        "The complexity index is a descriptive formula made from visible capped counts; it is not a learned risk score."
    )
    st.divider()
    st.subheader("Analyze against historical evidence")
    st.write(
        "Use the entered criteria to retrieve semantic comparables and calculate the current experimental model score."
    )
    if st.button("Find comparables and calculate score", type="primary"):
        if not text.strip():
            st.error("Enter eligibility criteria first.")
        else:
            try:
                tfidf, semantic = load_retrievers()
                model = load_best_uncalibrated_model()
                comparables = pd.DataFrame(semantic.search(text, k=5))
                model_row = pd.DataFrame([{**features, "eligibility_criteria": text}])
                score = float(model.predict_proba(model_row)[0, 1])
                local_explanation = explain_linear_prediction(model, model_row, top_n=10)
                st.metric(
                    "Termination-associated model score", f"{score:.3f} / 1.000", border=True
                )
                st.warning(
                    "This is an uncalibrated retrospective model score—not a real-world "
                    "termination probability, recommendation, or causal assessment."
                )
                st.dataframe(
                    comparables[
                        [
                            "rank",
                            "nct_id",
                            "brief_title",
                            "query_conditions",
                            "overall_status",
                            "similarity",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )
                top_candidate = str(comparables.iloc[0]["nct_id"])
                overlap = tfidf.explain_text_candidate(text, top_candidate, top_n=10)
                if overlap:
                    st.write(
                        "**Lexical evidence shared with the top semantic result:** "
                        + ", ".join(item["term"] for item in overlap)
                    )
                    st.caption(
                        "These TF-IDF overlaps are a transparent post-hoc aid; they are not the internal explanation of MiniLM."
                    )
                if not local_explanation.empty:
                    st.subheader("What moved the experimental model score")
                    explanation_plot = local_explanation.sort_values("contribution")
                    explanation_plot["direction"] = explanation_plot["contribution"].map(
                        lambda value: "toward terminated" if value > 0 else "toward completed"
                    )
                    fig = px.bar(
                        explanation_plot,
                        x="contribution",
                        y="feature",
                        color="direction",
                        orientation="h",
                        color_discrete_map={
                            "toward terminated": "#e26a45",
                            "toward completed": "#2878d0",
                        },
                    )
                    st.plotly_chart(fig, width="stretch")
                    st.caption(
                        "Contributions explain this linear model's score; they are statistical associations, not causes."
                    )
            except FileNotFoundError:
                st.error("Retrieval or model artifacts are missing; rebuild the local pipeline.")


def retrieval_page(frame: pd.DataFrame) -> None:
    header(
        "Comparable historical trials",
        "Compare lexical TF-IDF retrieval with sentence-transformer semantic retrieval",
    )
    try:
        tfidf, semantic = load_retrievers()
    except FileNotFoundError:
        st.error(
            "Retrieval artifacts are missing. Run `python scripts/build_retrieval_indexes.py`."
        )
        return
    labels = {
        f"{row.nct_id} — {row.brief_title}": row.nct_id
        for row in frame[["nct_id", "brief_title"]].itertuples(index=False)
    }
    chosen_label = st.selectbox("Choose a query trial", list(labels))
    method = st.radio("Retrieval representation", ["Semantic", "TF-IDF"], horizontal=True)
    k = st.slider("Number of comparable trials", 3, 10, 5)
    query_id = labels[chosen_label]
    retriever = semantic if method == "Semantic" else tfidf
    results = retriever.search_by_id(query_id, k=k)
    result_frame = pd.DataFrame(results)
    display_columns = [
        "rank",
        "nct_id",
        "brief_title",
        "query_conditions",
        "phases",
        "overall_status",
        "similarity",
    ]
    st.dataframe(
        result_frame[display_columns],
        width="stretch",
        hide_index=True,
        column_config={"similarity": st.column_config.NumberColumn(format="%.3f")},
    )
    with st.expander("Inspect the top result's eligibility text"):
        if results:
            st.write(results[0]["eligibility_criteria"])
    if results:
        overlap = tfidf.explain_pair(query_id, results[0]["nct_id"], top_n=10)
        st.subheader("Why the top result looks comparable")
        context = frame.set_index("nct_id").loc[[query_id, results[0]["nct_id"]]]
        context = context[
            ["brief_title", "query_conditions", "phases", "complexity_index", "criterion_count"]
        ].reset_index()
        st.dataframe(context, width="stretch", hide_index=True)
        if overlap:
            st.write("**Shared weighted terms:** " + ", ".join(item["term"] for item in overlap))
        if method == "Semantic":
            st.caption(
                "Shared TF-IDF terms are post-hoc lexical evidence. MiniLM similarity itself is based on dense semantic embeddings."
            )
    st.caption(
        "Similarity ranks historical text; it does not imply that one trial caused or predicts another trial's outcome."
    )
    proxy_path = RETRIEVAL_DIR / "proxy_evaluation.json"
    if proxy_path.exists():
        proxy = json.loads(proxy_path.read_text(encoding="utf-8"))
        st.subheader("Weak-label sanity check")
        proxy_frame = pd.DataFrame(proxy["metrics"]).T.reset_index(names="method")
        st.dataframe(proxy_frame, width="stretch", hide_index=True)
        st.caption(proxy["warning"])


def model_page() -> None:
    header(
        "Model evidence",
        "Chronological validation, calibration evidence, and global explanations",
    )
    try:
        (
            comparison,
            reliability,
            importance,
            subgroup,
            validation_comparison,
            summary,
            text_leakage_audit,
        ) = load_model_outputs()
    except FileNotFoundError:
        st.error("Model artifacts are missing. Run `python scripts/train_models.py`.")
        return
    columns = st.columns(4)
    columns[0].metric("Training trials", f"{summary['records_train']:,}", border=True)
    columns[1].metric("Temporal holdout", f"{summary['records_test']:,}", border=True)
    columns[2].metric("Cutoff", summary["temporal_cutoff"], border=True)
    columns[3].metric(
        "Best experiment",
        summary["best_uncalibrated_model"].replace("__", " / "),
        border=True,
    )
    st.caption(
        f"Model selection used {summary['records_selection_train']:,} oldest trials "
        f"for fitting and {summary['records_validation']:,} subsequent trials from "
        f"{summary['validation_cutoff']}; the selected model was refitted on all "
        f"{summary['records_train']:,} pre-test trials."
    )
    st.info(
        "Lexical leakage control removed exact registry-outcome words from eligibility text "
        f"inside the NLP pipeline: {text_leakage_audit['raw']['records_with_any_term']} "
        "raw matches, "
        f"{text_leakage_audit['sanitized_model_text']['records_with_any_term']} "
        "remaining before model fitting and inference."
    )

    plot_frame = comparison.melt(
        id_vars=["model", "family", "mode"],
        value_vars=["average_precision", "roc_auc"],
        var_name="metric",
        value_name="score",
    )
    fig = px.bar(
        plot_frame,
        x="model",
        y="score",
        color="metric",
        barmode="group",
        range_y=[0.45, 0.8],
        title="Model comparison on the latest 20% of trials",
    )
    st.plotly_chart(fig, width="stretch")
    st.dataframe(comparison.round(3), width="stretch", hide_index=True)
    strategy_frame = pd.DataFrame(
        [
            {"split": name, **values}
            for name, values in validation_comparison.items()
            if name in {"chronological", "random_stratified"}
        ]
    )
    st.subheader("Validation strategy sensitivity")
    st.dataframe(
        strategy_frame[
            [
                "split",
                "train_records",
                "test_records",
                "average_precision",
                "roc_auc",
                "brier_score",
            ]
        ].round(3),
        width="stretch",
        hide_index=True,
    )
    st.caption(validation_comparison["warning"])

    left, right = st.columns(2)
    with left:
        calibration = reliability.dropna(subset=["mean_predicted_probability"])
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=calibration["mean_predicted_probability"],
                y=calibration["observed_terminated_fraction"],
                mode="lines+markers",
                name="Observed",
            )
        )
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Ideal", line={"dash": "dash"})
        )
        fig.update_layout(
            title="Sigmoid calibration on temporal holdout",
            xaxis_title="Mean predicted probability",
            yaxis_title="Observed terminated fraction",
        )
        st.plotly_chart(fig, width="stretch")
    with right:
        top = importance.head(15).sort_values("absolute_importance")
        fig = px.bar(
            top,
            x="absolute_importance",
            y="feature",
            orientation="h",
            title="Top global model signals",
        )
        st.plotly_chart(fig, width="stretch")

    uncalibrated = summary["best_uncalibrated_metrics"]
    calibrated = summary["calibrated_test_metrics"]
    delta = summary["nlp_vs_structured_average_precision"]["logistic_regression"]
    st.info(
        "For Logistic Regression, NLP minus structured average precision was "
        f"{delta['mean_ap_difference']:+.3f} (paired-bootstrap 95% CI "
        f"{delta['ci_95_lower']:+.3f} to {delta['ci_95_upper']:+.3f})."
    )
    if calibrated["brier_score"] >= uncalibrated["brier_score"]:
        st.warning(
            "Calibration did not improve the temporal holdout "
            f"(Brier {uncalibrated['brier_score']:.3f} → {calibrated['brier_score']:.3f}). "
            "ProtocolLens reports this negative result and does not claim population-calibrated risk."
        )
    st.subheader("Therapeutic-area error analysis")
    st.dataframe(subgroup.round(3), width="stretch", hide_index=True)
    weakest = subgroup.sort_values("average_precision").iloc[0]
    st.warning(
        f"Performance is not uniform: {weakest['query_conditions']} has the "
        f"lowest subgroup average precision ({weakest['average_precision']:.3f}). "
        "This argues for broader data, area-specific error review, and external validation."
    )
    st.caption(summary["probability_scope"])


def retrieval_review_page() -> None:
    header(
        "Retrieval relevance review",
        "Apply one rubric consistently to TF-IDF and semantic results",
    )
    if not REVIEW_PATH.exists():
        st.error("Review sheet missing. Run `python scripts/create_retrieval_evaluation.py`.")
        return
    st.markdown(
        """
        Rate each query/candidate pair using the title, area, and protocol context:

        - **0 — irrelevant:** not useful as a comparable trial
        - **1 — partly relevant:** same broad topic but important population/design differences
        - **2 — highly relevant:** strongly comparable disease, population, and design context
        """
    )
    review = pd.read_csv(REVIEW_PATH)
    required_context = {
        "query_phase",
        "query_intervention_types",
        "query_eligibility",
        "candidate_phase",
        "candidate_intervention_types",
        "candidate_eligibility",
    }
    if not required_context.issubset(review.columns):
        st.error(
            "The review sheet predates the focused review workflow. Run "
            "`python scripts/create_retrieval_evaluation.py` once to refresh it."
        )
        return
    review["relevance_0_2"] = pd.to_numeric(review["relevance_0_2"], errors="coerce")
    review["reviewer_note"] = review["reviewer_note"].fillna("").astype(str)
    review_sources = (
        set(review["review_source"].dropna().astype(str))
        if "review_source" in review.columns
        else set()
    )
    ai_assisted = review_sources == {"ai_assisted_llm"}
    if ai_assisted:
        st.info(
            "Current labels are AI-assisted qualitative judgments. They are useful for "
            "error analysis but are not an independent human gold standard."
        )
    review = (
        review.sort_values(["query_nct_id", "method", "rank", "candidate_nct_id"])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    st.caption(
        "Retrieval method, candidate outcome, and similarity score are hidden while judging to reduce bias."
    )
    labelled, completed_pairs, total_pairs = review_progress(review)
    st.progress(
        completed_pairs / total_pairs,
        text=(
            f"{completed_pairs} of {total_pairs} unique comparisons complete · "
            f"{labelled} of {len(review)} method-level rows labeled"
        ),
    )

    pending = review.index[review["relevance_0_2"].isna()].tolist()
    if pending:
        current_index = pending[0]
        current = review.loc[current_index]
        comparison_number = completed_pairs + 1
        st.subheader(f"Blinded comparison {comparison_number} of {total_pairs}")
        st.caption(
            "Compare disease, population, phase, intervention context, and eligibility. "
            "The next unrated pair appears automatically after saving."
        )
        query_col, candidate_col = st.columns(2)
        with query_col:
            st.markdown("#### Query protocol")
            st.markdown(f"**{current['query_title']}**")
            st.caption(
                f"{current['query_nct_id']} · {current['query_area']} · "
                f"Phase: {display_text(current['query_phase'])} · "
                "Intervention: "
                f"{display_text(current['query_intervention_types'])}"
            )
            st.text_area(
                "Query eligibility criteria",
                value=display_text(current["query_eligibility"]),
                height=280,
                disabled=True,
                key=f"query_context_{current_index}",
            )
        with candidate_col:
            st.markdown("#### Candidate protocol")
            st.markdown(f"**{current['candidate_title']}**")
            st.caption(
                f"{current['candidate_nct_id']} · {current['candidate_area']} · "
                f"Phase: {display_text(current['candidate_phase'])} · "
                "Intervention: "
                f"{display_text(current['candidate_intervention_types'])}"
            )
            st.text_area(
                "Candidate eligibility criteria",
                value=display_text(current["candidate_eligibility"]),
                height=280,
                disabled=True,
                key=f"candidate_context_{current_index}",
            )

        rating_labels = {
            0: "0 — Irrelevant",
            1: "1 — Partly relevant",
            2: "2 — Highly relevant",
        }
        review_key = f"{current['query_nct_id']}_{current['method']}_{current['candidate_nct_id']}"
        note = st.text_input(
            "Reviewer note (optional)",
            value=str(current["reviewer_note"]),
            key=f"note_{review_key}",
        )
        st.markdown("**Choose one rating to save and advance:**")
        rating_columns = st.columns(3)
        selected_rating = None
        for rating, column in zip(rating_labels, rating_columns, strict=True):
            if column.button(
                rating_labels[rating],
                key=f"rating_{rating}_{review_key}",
                width="stretch",
            ):
                selected_rating = rating
        if selected_rating is not None:
            review = apply_pair_judgment(
                review,
                str(current["query_nct_id"]),
                str(current["candidate_nct_id"]),
                selected_rating,
                note,
            )
            save_review_progress(review, REVIEW_PATH)
            st.rerun()

    blinded_columns = [
        "query_nct_id",
        "query_area",
        "query_title",
        "rank",
        "candidate_nct_id",
        "candidate_area",
        "candidate_title",
        "relevance_0_2",
        "reviewer_note",
    ]
    with st.expander("Review progress table"):
        st.dataframe(review[blinded_columns], width="stretch", hide_index=True)
    blinded_export = review.drop(columns=["method", "candidate_status", "similarity"])
    st.download_button(
        "Download blinded review CSV",
        data=blinded_export.to_csv(index=False).encode("utf-8-sig"),
        file_name="retrieval_evaluation_blinded.csv",
        mime="text/csv",
    )

    if labelled == len(review):
        summary = summarize_judgments(review.to_dict(orient="records"), k=5)
        result = {
            "review_type": (
                "AI-assisted relevance judgments"
                if ai_assisted
                else "manual human relevance judgments"
            ),
            "review_source": "ai_assisted_llm" if ai_assisted else "manual_human",
            "independent_human_review": not ai_assisted,
            "judgments": len(review),
            "k": 5,
            "metrics": summary,
        }
        if ai_assisted:
            result["limitation"] = (
                "These judgments are an LLM-based qualitative assessment and are not "
                "an independent human gold standard."
            )
        manual_path = RETRIEVAL_DIR / "manual_evaluation.json"
        manual_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        st.success("Relevance review complete. The evaluation artifact has been saved.")
        metric_rows = [{"method": method, **metrics} for method, metrics in summary.items()]
        st.subheader("AI-assisted relevance metrics" if ai_assisted else "Human relevance metrics")
        st.dataframe(pd.DataFrame(metric_rows).round(4), width="stretch", hide_index=True)


def limitations_page() -> None:
    header("Method and limitations", "What ProtocolLens can—and cannot—support")
    st.markdown(
        """
        ### What is implemented

        - ClinicalTrials.gov API v2 ingestion with pagination, retries, raw-data retention, and metadata.
        - Inclusion/exclusion parsing and an interpretable complexity profile.
        - TF-IDF and sentence-transformer comparable-trial retrieval.
        - Structured-only, NLP-only, and combined Logistic Regression experiments.
        - Chronological holdout, probability calibration test, and global feature explanations.

        ### Critical limitations

        - ClinicalTrials.gov exposes the current study record. Eligibility text may have been edited
          after trial launch, so a chronological split alone cannot prove true prospective validity.
        - Completed versus terminated is a registry-status classification task, not clinical efficacy.
        - The cohort deliberately balances labels, so probabilities do not estimate population prevalence.
        - Independent retrieval validation requires human judgments; shared-condition labels and
          AI-assisted ratings are supporting evidence, not a human gold standard.
        - Associations and feature importance must not be interpreted as causes or medical advice.
        """
    )


def main() -> None:
    if not DATA_PATH.exists():
        st.error("Dataset missing. Run the ingestion and feature scripts first.")
        return
    frame = load_trials()
    st.sidebar.title("ProtocolLens")
    page = st.sidebar.radio(
        "Navigate",
        [
            "Overview",
            "Eligibility analyzer",
            "Comparable trials",
            "Retrieval review",
            "Model evidence",
            "Method & limitations",
        ],
    )
    st.sidebar.caption("Research demonstration · not medical advice")
    if page == "Overview":
        overview_page(frame)
    elif page == "Eligibility analyzer":
        eligibility_page(frame)
    elif page == "Comparable trials":
        retrieval_page(frame)
    elif page == "Retrieval review":
        retrieval_review_page()
    elif page == "Model evidence":
        model_page()
    else:
        limitations_page()


if __name__ == "__main__":
    main()
