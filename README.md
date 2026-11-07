# ICU Admission Prediction Using Machine Learning

An end-to-end healthcare Machine Learning project to predict the likelihood of ICU admission using patient clinical information.

---

## 📁 Project Structure

```text
ICU_Admission_Prediction/
│
├── data/
│   ├── raw/                 # Original, untouched clinical datasets
│   └── processed/           # Cleaned, encoded, and transformed datasets
│
├── notebooks/               # Jupyter notebooks for EDA and rapid prototyping
│
├── src/                     # Modular source code for the ML pipeline
│   ├── __init__.py          # Package initializer
│   ├── data_loader.py       # Data loading and initial inspection
│   ├── preprocessing.py     # Missing value handling, encoding, feature scaling
│   ├── train.py             # Model training and hyperparameter tuning
│   ├── evaluate.py          # Metrics, confusion matrix, ROC-AUC evaluation
│   └── predict.py           # Inference pipeline for new patient observations
│
├── models/                  # Saved model artifacts (.pkl, .joblib) & encoders
│
├── app/                     # Streamlit web application
│   └── app.py               # Interactive user interface for clinicians/users
│
├── .gitignore               # Ignored files (virtual environments, cache, checkpoints)
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 🚀 Setup Instructions

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
