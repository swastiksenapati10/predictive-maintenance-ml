"""
Predictive Maintenance & Fault Diagnostics Dashboard
Student: Swastik Senapati | Reg. No.: 2402060006

Run from the project root:
    streamlit run app.py

IMPORTANT:
- Fault classifier is trained on simulated sensor-feature data.
- NASA RUL model is trained on NASA C-MAPSS FD001 turbofan data.
- These are separate demonstration pipelines, not one validated motor-monitoring system.
"""

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
DATA_DIR_CANDIDATES = [
    ROOT / "data" / "nasa_cmapss",
    ROOT / "nasa_cmapss",
    ROOT,
]
DATA_DIR = next(
    (candidate for candidate in DATA_DIR_CANDIDATES
     if (candidate / "train_FD001.txt").exists()),
    ROOT / "data" / "nasa_cmapss",
)

st.set_page_config(
    page_title="Predictive Maintenance | Swastik Senapati",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------- Styling --------------------
st.markdown(
    """
    <style>
    .main { background-color: #f6f8fb; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .hero {
        padding: 1.25rem 1.5rem; border-radius: 16px;
        background: linear-gradient(110deg, #102a43, #176b87);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 { color: white; margin: 0; font-size: 2rem; }
    .hero p { color: #d9eaf2; margin: .35rem 0 0 0; }
    .note {
        background: #fff8e6; border-left: 4px solid #e0a126;
        padding: .8rem 1rem; border-radius: 8px; color: #594214;
    }
    .small-muted { color: #64748b; font-size: .85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------- Helpers --------------------
@st.cache_resource(show_spinner=False)
def load_fault_package(path_string, modified_time):
    return joblib.load(path_string)

@st.cache_resource(show_spinner=False)
def load_rul_package(path_string, modified_time):
    return joblib.load(path_string)

@st.cache_data(show_spinner=False)
def read_csv_cached(path_string, modified_time):
    return pd.read_csv(path_string)

@st.cache_data(show_spinner=False)
def read_nasa_txt(path_string, modified_time):
    columns = (
        ["engine_id", "cycle"]
        + [f"setting_{i}" for i in range(1, 4)]
        + [f"sensor_{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(path_string, sep=r"\s+", header=None)
    df.columns = columns
    return df

def file_stamp(path):
    return path.stat().st_mtime if path.exists() else 0

def metric_card(label, value, help_text=None):
    st.metric(label, value, help=help_text)

def show_missing_file(path, instruction):
    st.warning(f"Required file not found: `{path.relative_to(ROOT)}`")
    st.caption(instruction)

# -------------------- Sidebar --------------------
st.sidebar.title("⚙️ Project Navigation")
page = st.sidebar.radio(
    "Select dashboard section",
    ["Overview", "Fault Diagnostics", "NASA RUL Prediction", "Model Evaluation", "About"],
)
st.sidebar.divider()
st.sidebar.caption("AI & ML Project")
st.sidebar.write("**Swastik Senapati**")
st.sidebar.write("Registration No.: `2402060006`")
st.sidebar.caption("Electrical & Electronics Engineering")

# -------------------- Header --------------------
st.markdown(
    """
    <div class="hero">
      <h1>Predictive Maintenance & Fault Diagnostics</h1>
      <p>Machine-learning prototype • Fault classification • Remaining Useful Life estimation</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Resolve model paths
fault_model_path = ROOT / "fault_diagnosis_model.joblib"
sample_path = ROOT / "sample_input.csv"

rul_model_candidates = [
    MODELS / "best_nasa_rul_model.joblib",
    MODELS / "nasa_fd001_rul_model.joblib",
]
rul_model_path = next((p for p in rul_model_candidates if p.exists()), None)

# -------------------- Overview --------------------
if page == "Overview":
    st.subheader("System Overview")
    st.markdown(
        """
        This dashboard presents two model pipelines developed for the project.
        The first classifies simulated machine sensor-feature patterns. The second
        estimates turbofan-engine Remaining Useful Life using NASA C-MAPSS FD001.
        """
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Fault model", "Random Forest", "Trained on simulated feature data")
    with c2:
        metric_card("RUL model", "NASA FD001", "Turbofan engine degradation dataset")
    with c3:
        metric_card("Dashboard", "Streamlit", "Local interactive demonstration")

    st.subheader("Available project artifacts")
    artifact_rows = [
        ("Fault classifier", fault_model_path),
        ("Sample fault input", sample_path),
        ("NASA FD001 training data", DATA_DIR / "train_FD001.txt"),
        ("NASA FD001 test data", DATA_DIR / "test_FD001.txt"),
        ("NASA FD001 RUL targets", DATA_DIR / "RUL_FD001.txt"),
        ("NASA RUL model", rul_model_path),
    ]
    status_data = []
    for label, path in artifact_rows:
        status_data.append({
            "Artifact": label,
            "Status": "Available" if path is not None and path.exists() else "Not found",
        })
    st.dataframe(pd.DataFrame(status_data), use_container_width=True, hide_index=True)

    st.markdown(
        """
        <div class="note">
        <b>Interpretation and safety note:</b> The fault classifier currently uses
        simulated feature data. The NASA RUL model uses turbofan-engine data.
        These are separate research demonstrations and must not be treated as a
        validated real-time electrical-motor protection or maintenance system.
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------- Fault Diagnostics --------------------
elif page == "Fault Diagnostics":
    st.subheader("Fault Diagnostics")
    st.write("Classify one observation using the existing Random Forest fault classifier.")

    if not fault_model_path.exists():
        show_missing_file(fault_model_path, "Place `fault_diagnosis_model.joblib` in the project root.")
        st.stop()

    try:
        fault_package = load_fault_package(
            str(fault_model_path), file_stamp(fault_model_path)
        )
        fault_model = fault_package["model"]
        feature_columns = fault_package["feature_columns"]
        class_names = fault_package.get("classes", list(fault_model.classes_))
    except Exception as exc:
        st.error(f"Could not load fault model: {exc}")
        st.stop()

    st.info(
        "Use the bundled sample input for a demonstration, or upload a CSV containing "
        "the same numeric feature columns used during training."
    )
    uploaded = st.file_uploader("Upload sensor-feature CSV (optional)", type=["csv"], key="fault_csv")

    try:
        if uploaded is not None:
            input_df = pd.read_csv(uploaded)
            source_label = "Uploaded CSV"
        elif sample_path.exists():
            input_df = pd.read_csv(sample_path)
            source_label = "Bundled sample_input.csv"
        else:
            st.error("sample_input.csv was not found and no CSV was uploaded.")
            st.stop()

        missing = [col for col in feature_columns if col not in input_df.columns]
        if missing:
            st.error("Your CSV is missing required feature columns: " + ", ".join(missing))
            st.stop()

        if input_df.empty:
            st.error("The uploaded CSV contains no rows.")
            st.stop()

        row_number = st.number_input(
            "Select row number (starting at 1)",
            min_value=1,
            max_value=len(input_df),
            value=1,
            step=1,
        )
        row = input_df[feature_columns].iloc[[int(row_number) - 1]]

        st.caption(f"Input source: {source_label}")
        with st.expander("View model input features"):
            st.dataframe(row, use_container_width=True)

        if st.button("Run fault diagnosis", type="primary"):
            prediction = fault_model.predict(row)[0]
            probabilities = fault_model.predict_proba(row)[0]
            best_probability = float(np.max(probabilities))

            left, right = st.columns([1, 1])
            with left:
                st.metric("Predicted condition", str(prediction))
            with right:
                st.metric("Highest class probability", f"{best_probability * 100:.2f}%")

            prob_df = pd.DataFrame({
                "Fault class": fault_model.classes_,
                "Model probability": probabilities,
            }).sort_values("Model probability", ascending=False)

            st.subheader("Class probability distribution")
            st.bar_chart(prob_df.set_index("Fault class"))
            st.dataframe(
                prob_df.assign(**{"Model probability": prob_df["Model probability"].map(lambda x: f"{x:.4f}")}),
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                "Probabilities are model scores for this simulated-data classifier; "
                "they are not calibrated real-world fault probabilities."
            )
    except Exception as exc:
        st.error(f"Fault diagnosis could not be completed: {exc}")

# -------------------- NASA RUL Prediction --------------------
elif page == "NASA RUL Prediction":
    st.subheader("NASA C-MAPSS FD001 — Remaining Useful Life")
    st.write(
        "Select a test engine. The dashboard takes its latest observed sensor row "
        "and estimates RUL in operating cycles."
    )

    test_path = DATA_DIR / "test_FD001.txt"
    target_path = DATA_DIR / "RUL_FD001.txt"

    if rul_model_path is None:
        st.warning(
            "No NASA model was found. Run `python nasa_rul_model.py` first. "
            "Expected: `models/nasa_fd001_rul_model.joblib`."
        )
        st.stop()
    if not test_path.exists() or not target_path.exists():
        st.warning("NASA FD001 test data or RUL target file is missing from data/nasa_cmapss.")
        st.stop()

    try:
        rul_package = load_rul_package(
            str(rul_model_path), file_stamp(rul_model_path)
        )
        rul_model = rul_package["model"]
        rul_features = rul_package["features"]

        test_df = read_nasa_txt(str(test_path), file_stamp(test_path))
        target_df = pd.read_csv(target_path, sep=r"\s+", header=None)
        actual_targets = target_df.iloc[:, 0].to_numpy()

        engine_ids = sorted(test_df["engine_id"].unique())
        selected_engine = st.selectbox("Select NASA test engine ID", engine_ids, index=0)

        engine_rows = test_df[test_df["engine_id"] == selected_engine].sort_values("cycle")
        latest_row = engine_rows.iloc[[-1]].copy()

        missing = [col for col in rul_features if col not in latest_row.columns]
        if missing:
            st.error("The loaded model expects unavailable features: " + ", ".join(missing))
            st.stop()

        model_input = latest_row[rul_features]
        prediction = max(0.0, float(rul_model.predict(model_input)[0]))

        # NASA target file order corresponds to sorted test engine IDs.
        target_index = engine_ids.index(selected_engine)
        actual_rul = float(actual_targets[target_index])

        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Predicted RUL", f"{prediction:.1f} cycles")
        with c2:
            metric_card("NASA reference RUL", f"{actual_rul:.1f} cycles")
        with c3:
            metric_card("Absolute error", f"{abs(prediction - actual_rul):.1f} cycles")

        st.subheader(f"Engine {selected_engine} — latest observed measurements")
        st.write(
            f"Latest observed cycle: **{int(latest_row['cycle'].iloc[0])}**  |  "
            f"Observed windows: **{len(engine_rows)}**"
        )

        sensor_cols = [f"sensor_{i}" for i in range(1, 22)]
        visible_cols = [c for c in ["cycle", "setting_1", "setting_2", "setting_3"] + sensor_cols if c in latest_row.columns]
        with st.expander("View latest sensor measurements"):
            st.dataframe(latest_row[visible_cols], use_container_width=True)

        st.markdown(
            """
            <div class="note">
            The NASA reference RUL is shown only for benchmark evaluation. It is not
            supplied to the model as an input. This is a turbofan-engine benchmark,
            not a direct estimate of electrical-motor life.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Show this engine's observed sensor trends.
        selected_sensor = st.selectbox(
            "Choose sensor for trend plot",
            ["sensor_2", "sensor_3", "sensor_4", "sensor_7", "sensor_11", "sensor_12", "sensor_15", "sensor_20", "sensor_21"],
            index=0,
        )
        if selected_sensor in engine_rows.columns:
            trend = engine_rows[["cycle", selected_sensor]].set_index("cycle")
            st.line_chart(trend, use_container_width=True)

    except Exception as exc:
        st.error(f"NASA RUL prediction could not be completed: {exc}")

# -------------------- Model Evaluation --------------------
elif page == "Model Evaluation":
    st.subheader("Model Evaluation")

    metrics_path = OUTPUTS / "nasa_fd001_metrics.csv"
    comparison_path = OUTPUTS / "model_comparison.csv"
    predictions_path = OUTPUTS / "nasa_fd001_predictions.csv"

    if metrics_path.exists():
        metrics_df = read_csv_cached(str(metrics_path), file_stamp(metrics_path))
        st.markdown("### NASA FD001 test-set metrics")
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
        if not metrics_df.empty:
            row = metrics_df.iloc[0]
            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("MAE", f"{row['MAE']:.2f} cycles")
            with c2:
                metric_card("RMSE", f"{row['RMSE']:.2f} cycles")
            with c3:
                metric_card("R²", f"{row['R2']:.4f}")
    else:
        st.info("Run `python nasa_rul_model.py` to generate NASA FD001 metrics.")

    if comparison_path.exists():
        st.markdown("### Algorithm comparison")
        comparison_df = read_csv_cached(str(comparison_path), file_stamp(comparison_path))
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        if "Model" in comparison_df.columns and "MAE" in comparison_df.columns:
            st.bar_chart(comparison_df.set_index("Model")[["MAE"]])
    else:
        st.info("Optional: run `python compare_models.py` to generate the model comparison table.")

    if predictions_path.exists():
        pred_df = read_csv_cached(str(predictions_path), file_stamp(predictions_path))
        st.markdown("### NASA test predictions")
        st.dataframe(pred_df.head(20), use_container_width=True, hide_index=True)
        if {"actual_RUL", "predicted_RUL"}.issubset(pred_df.columns):
            chart_df = pred_df[["actual_RUL", "predicted_RUL"]].reset_index(drop=True)
            chart_df.index.name = "Test engine row"
            st.line_chart(chart_df, use_container_width=True)

# -------------------- About --------------------
elif page == "About":
    st.subheader("Project Information")
    st.write("**Project title:** Machine Learning for Predictive Maintenance and Fault Diagnostics in Industrial Systems")
    st.write("**Student:** Swastik Senapati")
    st.write("**Registration number:** 2402060006")
    st.write("**Department:** Electrical and Electronics Engineering")
    st.write("**Fault diagnosis algorithm:** Random Forest Classifier")
    st.write("**RUL algorithm:** Random Forest Regressor (NASA FD001 baseline or selected comparison model)")
    st.divider()
    st.markdown(
        """
        **Current limitations**
        - Fault classification uses a synthetic demonstration dataset.
        - NASA C-MAPSS represents turbofan engines, not electrical motors.
        - The two models currently use separate datasets and input pipelines.
        - This dashboard is an academic prototype, not a safety-certified monitoring system.
        - Model outputs should not be used to control or protect real machinery.
        """
    )

st.sidebar.divider()
st.sidebar.caption("Academic prototype • Not for safety-critical use")
