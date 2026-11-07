from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from matplotlib.patches import Wedge


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "icu_logistic_model.pkl"

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "icu_prediction_data.csv"

COMPARISON_PATH = PROJECT_ROOT / "data" / "processed" / "model_comparison.png"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="ICU Admission Prediction System",
    page_icon="🏥",
    layout="wide"
)


# =========================================================
# HEALTHCARE DASHBOARD THEME
# =========================================================

st.markdown("""
<style>

.stApp {
    background-color: #f1f5fb;
    color: #1e293b;
}

[data-testid="stMain"] {
    background-color: #f1f5fb;
}

h1 {
    color: #12345a !important;
    font-weight: 750 !important;
}

h2 {
    color: #17456e !important;
    font-weight: 700 !important;
}

h3 {
    color: #176b87 !important;
    font-weight: 650 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #26364a;
}

[data-testid="stMetric"] {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 14px;
    border: 1px solid #d6e1ed;
    box-shadow: 0 3px 12px rgba(30, 64, 100, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stMetricValue"] {
    color: #087e9b !important;
    font-weight: 750 !important;
}

.stButton > button {
    background-color: #087e9b;
    color: white !important;
    border-radius: 9px;
    border: none;
    padding: 8px 18px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #075985;
    color: white !important;
}

.stDownloadButton > button {
    background-color: #164e63;
    color: white !important;
    border-radius: 9px;
}

button[data-baseweb="tab"] {
    color: #17456e !important;
    font-weight: 600;
}

[data-testid="stSidebar"] {
    background-color: #e1ebf5;
}

[data-testid="stSidebar"] * {
    color: #17365d;
}

[data-testid="stDataFrame"] {
    background-color: white;
    border: 1px solid #d6e1ed;
    border-radius: 10px;
}

[data-testid="stFileUploader"] {
    background-color: #ffffff;
    border: 1px dashed #087e9b;
    border-radius: 12px;
    padding: 12px;
}

[data-testid="stAlert"] {
    border-radius: 10px;
}

hr {
    border-color: #d6e1ed;
}

.disclaimer-box {
    background-color: #e3f2fd;
    color: #17365d;
    padding: 17px 20px;
    border: 1px solid #90caf9;
    border-left: 6px solid #1976d2;
    border-radius: 10px;
    font-size: 14px;
    line-height: 1.7;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 13px;
    padding: 20px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD TRAINED MODEL
# =========================================================

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()

except Exception as e:
    st.error(f"Unable to load the trained model: {e}")
    st.stop()


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


try:
    data = load_dataset()

except Exception as e:
    st.error(f"Unable to load the prediction dataset: {e}")
    st.stop()


# =========================================================
# FEATURE PREPARATION
# =========================================================

DROP_COLUMNS = [
    "ICU_FUTURE",
    "ICU",
    "PATIENT_VISIT_IDENTIFIER",
    "WINDOW"
]


def prepare_input(input_data):

    input_data = input_data.copy()

    input_data = input_data.drop(
        columns=DROP_COLUMNS,
        errors="ignore"
    )

    expected_features = list(model.feature_names_in_)

    input_data = input_data.reindex(
        columns=expected_features
    )

    return input_data


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def make_prediction(input_data):

    prepared_data = prepare_input(input_data)

    predictions = model.predict(prepared_data)

    probabilities = model.predict_proba(prepared_data)

    classes = list(model.classes_)

    positive_index = classes.index(1)

    positive_probabilities = probabilities[:, positive_index]

    return predictions, positive_probabilities


# =========================================================
# RISK GAUGE
# =========================================================

def show_risk_gauge(probability):

    fig, ax = plt.subplots(figsize=(6, 3.5))

    probability = float(probability)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.2)

    ax.axis("off")

    background = Wedge(
        (0, 0),
        1,
        0,
        180,
        width=0.22,
        facecolor="#dbe4ee"
    )

    ax.add_patch(background)

    angle = probability * 180

    foreground = Wedge(
        (0, 0),
        1,
        0,
        angle,
        width=0.22,
        facecolor="#087e9b"
    )

    ax.add_patch(foreground)

    ax.text(
        0,
        0.15,
        f"{probability * 100:.1f}%",
        ha="center",
        fontsize=27,
        fontweight="bold",
        color="#12345a"
    )

    ax.text(
        0,
        -0.08,
        "ICU Admission Probability",
        ha="center",
        fontsize=11,
        color="#475569"
    )

    st.pyplot(fig)

    plt.close(fig)


# =========================================================
# FEATURE IMPORTANCE - CORRECTED VERSION
# =========================================================

def show_feature_importance():

    try:

        # Get final Logistic Regression classifier
        classifier = model.steps[-1][1]

        # Get preprocessing pipeline
        preprocessing = model[:-1]

        # Retrieve transformed feature names
        feature_names = preprocessing.get_feature_names_out()

        # Retrieve model coefficients
        coefficients = classifier.coef_[0]

        if len(feature_names) != len(coefficients):

            st.warning(
                "Feature names and model coefficients do not match."
            )
            return

        importance = pd.DataFrame({
            "Feature": feature_names,
            "Importance": coefficients
        })

        importance["Absolute Importance"] = (
            importance["Importance"].abs()
        )

        importance = importance.sort_values(
            by="Absolute Importance",
            ascending=False
        ).head(10)

        st.markdown("### Top 10 Influential Clinical Features")

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.barh(
            importance["Feature"],
            importance["Importance"]
        )

        ax.set_xlabel("Logistic Regression Coefficient")

        ax.set_ylabel("Clinical Features")

        ax.set_title("Top 10 Influential Features")

        ax.axvline(
            x=0,
            color="black",
            linewidth=0.8
        )

        ax.invert_yaxis()

        fig.tight_layout()

        st.pyplot(fig)

        plt.close(fig)

        st.caption(
            "Positive coefficients increase the model's predicted "
            "log-odds of ICU admission, while negative coefficients "
            "decrease them. These are associations, not causal effects."
        )

    except Exception as e:

        st.warning(
            f"Feature importance could not be generated: {e}"
        )


# =========================================================
# HEADER
# =========================================================

st.markdown("# 🏥 ICU Admission Prediction System")

st.markdown(
    "### Machine Learning-Based Healthcare Research Prototype"
)

st.write(
    "An interactive healthcare analytics application "
    "for predicting future ICU admission using clinical data."
)


# =========================================================
# DISCLAIMER
# =========================================================

st.markdown("""
<div class="disclaimer-box">

<strong>Educational Research Prototype:</strong>

This model was developed using a COVID-19 hospital dataset
and has not been clinically validated. It must not be used
for real-world diagnosis, treatment, or clinical decisions.

</div>
""", unsafe_allow_html=True)

st.markdown("---")


# =========================================================
# MODEL PERFORMANCE
# =========================================================

st.markdown("## 📊 Model Performance")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Accuracy", "62%")

col2.metric("Precision", "57.1%")

col3.metric("Recall", "72.7%")

col4.metric("F1-Score", "64%")

col5.metric("ROC-AUC", "0.615")


# =========================================================
# MODEL COMPARISON
# =========================================================

st.markdown("## 📈 Model Comparison")

if COMPARISON_PATH.exists():

    st.image(
        str(COMPARISON_PATH),
        use_container_width=True
    )

else:

    st.warning("Model comparison image was not found.")

st.markdown("---")


# =========================================================
# MODEL INTERPRETATION
# =========================================================

st.markdown("## 🔍 Model Interpretation")

show_feature_importance()

st.markdown("---")


# =========================================================
# PREDICTION SECTION
# =========================================================

st.markdown("## 🩺 ICU Admission Prediction")

tab1, tab2 = st.tabs([
    "👤 Single Patient Prediction",
    "📂 Batch CSV Prediction"
])


# =========================================================
# TAB 1 - SINGLE PATIENT PREDICTION
# =========================================================

with tab1:

    st.markdown("### Sample Patient Prediction")

    st.write(
        "Select a sample patient record from the processed dataset "
        "to demonstrate the prediction system."
    )

    if st.button("Generate Sample Prediction"):

        sample = data.drop(
            columns=DROP_COLUMNS,
            errors="ignore"
        ).head(1)

        try:

            predictions, probabilities = make_prediction(sample)

            prediction = int(predictions[0])

            probability = float(probabilities[0])

            st.markdown("---")

            st.markdown("### Prediction Results")

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                if prediction == 1:

                    st.error(
                        "Predicted Outcome: Future ICU Admission"
                    )

                else:

                    st.success(
                        "Predicted Outcome: No Future ICU Admission"
                    )

                st.metric(
                    "Predicted ICU Probability",
                    f"{probability * 100:.1f}%"
                )

                if probability < 0.40:

                    st.info("Model probability category: Low")

                elif probability < 0.70:

                    st.warning("Model probability category: Moderate")

                else:

                    st.error("Model probability category: High")

            with result_col2:

                show_risk_gauge(probability)

            st.markdown("### Patient Input Data")

            st.dataframe(
                sample,
                use_container_width=True
            )

            st.caption(
                "Risk categories are illustrative probability bands "
                "for this research prototype, not clinical risk thresholds."
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")


# =========================================================
# TAB 2 - BATCH CSV PREDICTION
# =========================================================

with tab2:

    st.markdown("### Batch Patient Prediction")

    st.write(
        "Upload a CSV file containing patient clinical information "
        "to generate predictions for multiple records."
    )

    uploaded_file = st.file_uploader(
        "Upload Patient CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(uploaded_file)

            st.markdown("### Uploaded Dataset Preview")

            st.dataframe(
                uploaded_data.head(),
                use_container_width=True
            )

            if st.button("Generate Batch Predictions"):

                predictions, probabilities = make_prediction(
                    uploaded_data
                )

                results = uploaded_data.copy()

                results["Predicted_ICU_Admission"] = predictions

                results["ICU_Admission_Probability"] = (
                    probabilities * 100
                ).round(2)

                results["Prediction_Category"] = [
                    "ICU Admission" if p == 1
                    else "No ICU Admission"
                    for p in predictions
                ]

                st.markdown("### Batch Prediction Results")

                st.dataframe(
                    results,
                    use_container_width=True
                )

                total_patients = len(results)

                predicted_icu = int(
                    (predictions == 1).sum()
                )

                predicted_non_icu = (
                    total_patients - predicted_icu
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Patients",
                    total_patients
                )

                col2.metric(
                    "Predicted ICU Admissions",
                    predicted_icu
                )

                col3.metric(
                    "Predicted Non-ICU",
                    predicted_non_icu
                )

                csv_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Prediction Results",
                    data=csv_data,
                    file_name="icu_prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(f"Batch prediction error: {e}")


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown("""
<div class="footer">

<strong>ICU Admission Prediction System</strong><br>

Machine Learning Healthcare Research Prototype<br>

MSc Healthcare Informatics and Data Analytics

</div>
""", unsafe_allow_html=True)