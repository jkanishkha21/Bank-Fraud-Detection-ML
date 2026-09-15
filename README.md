# 🏦 Bank Fraud Detection using Machine Learning

## 1. Project Overview

This project demonstrates an AI/ML-based bank transaction fraud detection system.

The system learns patterns from transaction data and produces a fraud-risk probability for a new transaction. A configurable decision threshold is then used to flag potentially fraudulent transactions for further review.

The project follows the Bank Fraud Detection project brief supplied for the AI internship. The brief describes fraud detection as a real-time financial problem involving highly imbalanced transaction data, explainability, threshold optimization, and an end-to-end application. It lists Python, scikit-learn, XGBoost, SHAP, Streamlit, FastAPI and Docker among the suggested technologies.

## 2. Problem Statement

A digital bank needs to identify suspicious transactions while reducing missed fraud cases and unnecessary false alerts.

The system should consider transaction and risk-related attributes such as:

- Transaction amount
- Timestamp/hour
- Transaction type
- Merchant category
- Country
- Device risk score
- IP risk score
- Account age
- Previous transaction activity

## 3. Project Objectives

- Build a supervised machine-learning fraud classifier.
- Handle class imbalance using class-weighted learning.
- Produce a fraud probability for each transaction.
- Select a decision threshold with emphasis on fraud recall.
- Evaluate the model using precision, recall, F1-score and ROC-AUC.
- Provide an interactive Streamlit application.
- Make the project reproducible with a synthetic dataset generator.

## 4. Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Random Forest
- Joblib
- Streamlit

## 5. Project Structure

```text
Bank_Fraud_Detection_Project/
│
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
│
├── data/
│   └── transactions.csv
│
├── models/
│   ├── fraud_pipeline.joblib
│   └── model_metadata.json
│
└── reports/
    └── metrics.json
```

The `data`, `models`, and `reports` files are generated when the project is run.

## 6. Dataset

For reproducibility, this repository includes a synthetic transaction generator instead of using real customer banking information.

Run:

```bash
python generate_dataset.py
```

This creates:

```text
data/transactions.csv
```

The synthetic dataset contains transaction characteristics and a generated `is_fraud` target.

## 7. Machine Learning Pipeline

### Data preprocessing

Numeric features are standardized using `StandardScaler`.

Categorical features are converted into machine-readable values using `OneHotEncoder`.

### Model

The project uses `RandomForestClassifier` with:

```text
class_weight="balanced"
```

This is used because fraud detection is an imbalanced classification problem.

### Threshold optimization

Instead of automatically using 0.50 as the fraud threshold, the training script evaluates thresholds from 0.10 to 0.90 and selects one using a score that emphasizes recall while retaining precision.

### Evaluation metrics

The following metrics are reported:

- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

## 8. How to Run

### Step 1 — Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

### Step 3 — Generate the dataset

```bash
python generate_dataset.py
```

### Step 4 — Train the model

```bash
python train_model.py
```

This creates:

```text
models/fraud_pipeline.joblib
models/model_metadata.json
reports/metrics.json
```

### Step 5 — Launch the Streamlit app

```bash
streamlit run app.py
```

The browser will open the interactive fraud detection dashboard.

## 9. Application Features

### Transaction Analysis

Users can enter:

- Amount
- Transaction hour
- Transaction type
- Merchant category
- Country
- Device risk score
- IP risk score
- Account age
- Previous transactions

The application returns:

- Fraud-risk probability
- Fraud/normal classification
- Suggested review action

### Dataset Overview

The dashboard also displays:

- Number of transactions
- Number of fraud cases
- Fraud rate
- Sample records
- Fraud distribution

## 10. Limitations

This is an internship demonstration project using synthetic data.

It should not be used to make real financial, banking, compliance, or customer decisions.

The project does not claim production-grade regulatory compliance, real-time banking integration, or deployment to a live banking infrastructure.

## 11. Future Enhancements

Possible improvements include:

- XGBoost-based classification
- SHAP explainability
- FastAPI real-time inference endpoint
- Docker containerization
- AWS deployment
- Real-time transaction streaming
- Model drift monitoring
- Autoencoder-based anomaly detection
- Graph-based fraud-ring detection
- Analyst alert dashboard

## 12. Internship Project Alignment

The project is based on the Bank Fraud Detection project option in the provided AI internship project guide.

The guide identifies the project as a beginner-intermediate fraud detection project and describes requirements around imbalanced data, fraud recall, false-positive control, explainability and real-time inference.

This implementation focuses on a realistic, reproducible core version suitable for demonstration and further extension.

## 13. Author

**Name:** [Your Full Name]

**Course:** Artificial Intelligence

**Organization:** GlowLogics

**GitHub:** [Add your GitHub profile/repository link]

## 14. Disclaimer

This project is developed for educational and internship purposes only. The dataset is synthetic and contains no real customer banking information.
