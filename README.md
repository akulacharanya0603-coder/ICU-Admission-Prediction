# 🏥 ICU Admission Prediction Using Machine Learning

An end-to-end healthcare Machine Learning project that predicts the likelihood of future ICU admission using patient clinical data.

## 📌 Project Overview

The ICU Admission Prediction System is an educational healthcare analytics application developed using Machine Learning classification algorithms.

The project aims to identify patients who may require future Intensive Care Unit (ICU) admission based on their baseline clinical information.

A publicly available COVID-19 hospital dataset was used to develop and evaluate the prediction models.

The trained model is integrated into an interactive Streamlit dashboard that supports individual patient predictions and batch CSV predictions.

## 🎯 Objectives

- Develop a Machine Learning-based ICU admission prediction system.
- Perform clinical dataset preprocessing and exploratory data analysis.
- Implement and compare Logistic Regression and Random Forest algorithms.
- Evaluate model performance using classification metrics.
- Develop an interactive healthcare prediction dashboard.
- Demonstrate an end-to-end healthcare Machine Learning workflow.

## 📂 Dataset

**Dataset:** Kaggle_Sirio_Libanes_ICU_Prediction

The dataset contains clinical information from COVID-19 patients, including laboratory measurements, vital signs, demographic information and disease-related features.

### Dataset Processing

- Selected baseline patient records.
- Excluded patients already admitted to ICU at baseline.
- Created a future ICU admission target.
- Removed features with more than 60% missing values.
- Applied missing-value imputation.
- Encoded categorical variables.
- Applied feature scaling.
- Used stratified train-test splitting.

Final processed dataset: 353 eligible patient records.

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| Scikit-learn | Machine Learning |
| Joblib | Model serialization |
| Streamlit | Interactive web application |

## 🤖 Machine Learning Models

Two classification algorithms were implemented:

1. Logistic Regression
2. Random Forest Classifier

Logistic Regression was selected as the primary model based on its overall performance across the evaluated classification metrics.

## 📊 Model Performance

| Evaluation Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| Accuracy | 62.0% | 56.3% |
| Precision | 57.1% | 52.3% |
| Recall | 72.7% | 69.7% |
| F1-Score | 64.0% | 59.7% |
| ROC-AUC | 0.615 | 0.620 |

**Selected Model:** Logistic Regression

The Logistic Regression model achieved better accuracy, precision, recall and F1-score, while Random Forest achieved a slightly higher ROC-AUC.

The results indicate moderate predictive performance and require further validation.

## 🖥️ Application Features

The Streamlit dashboard includes:

- Healthcare-themed interactive dashboard.
- Model performance metrics.
- Model comparison visualization.
- Feature importance visualization.
- Single-patient sample prediction.
- ICU admission probability visualization.
- Batch CSV prediction.
- Downloadable prediction results.

## 🔄 Project Workflow

1. Dataset Collection
2. Data Preprocessing
3. Exploratory Data Analysis
4. Feature Selection
5. Train-Test Splitting
6. Model Development
7. Model Evaluation
8. Model Saving
9. Streamlit Application Development
10. Prediction and Visualization

## 📁 Project Structure

```text
ICU-Admission-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   ├── raw/
│   │   └── Kaggle_Sirio_Libanes_ICU_Prediction.xls
│   │
│   ├── processed/
│   │   ├── icu_prediction_data.csv
│   │   └── model_comparison.png
│   │
│   └── icu_distribution.png
│
├── models/
│   ├── icu_logistic_model.pkl
│   └── icu_random_forest_model.pkl
│
├── notebooks/
│
├── src/
│   └── __init__.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/akulacharanya0603-coder/ICU-Admission-Prediction.git
```

### 2. Navigate to the Project Directory

```bash
cd ICU-Admission-Prediction
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Install Dependencies

```bash
.venv\Scripts\activate
```

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit Application

```bash
python -m streamlit run app/app.py
```

The application will open in your browser at:

`http://localhost:8501`

## ⚠️ Limitations

- The dataset is based on COVID-19 hospital patients.
- The eligible modeling dataset contains 353 patient records.
- Model performance is moderate.
- External validation has not been performed.
- The application is an educational research prototype.

## 🔒 Disclaimer

This project is intended exclusively for educational and research purposes.

The model has not been clinically validated and must not be used for real-world diagnosis, treatment recommendations or clinical decision-making.

## 👩‍💻 Author

**Akula Charanya**

MSc Healthcare Informatics and Data Analytics

GitHub: [akulacharanya0603-coder](https://github.com/akulacharanya0603-coder)

---

⭐ If you find this project interesting, feel free to explore the repository!