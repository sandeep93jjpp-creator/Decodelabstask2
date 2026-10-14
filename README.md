# 🛡️ Fraud & Anomaly Detection Pipeline

## DecodeLabs Data Science Internship | Python · Pandas · Scikit-learn · Imbalanced-learn

*The problem:* Financial institutions and e-commerce platforms struggle to detect fraudulent or risky transactions due to severe class imbalance, causing standard models to falsely predict every transaction as "Legitimate" to achieve artificially high accuracy. 

*The solution:* A leak-free, end-to-end Machine Learning pipeline using synthetic oversampling (SMOTE) wrapped inside cross-validation, feature scaling, and ensemble modeling to maximize detection across multiple order statuses.

---

## 📌 Project Overview

This project implements a leak-free, multiclass fraud detection pipeline designed to identify risky order anomalies across multiple discrete transaction statuses without reducing the target into a simple binary variable. 

By leveraging **SMOTE** oversampling strictly within cross-validation loops, **StandardScaler**, and ensemble modeling (**Random Forest vs. Logistic Regression**), the solution maximizes detection performance while discarding misleading accuracy metrics.

---

## 🚀 Key Results

| Challenge | Approach | Outcome |
| :--- | :--- | :--- |
| **Severe Class Imbalance** | Applied SMOTE inside `imblearn.pipeline.Pipeline` | Rare status classes oversampled safely without data leakage |
| **Target Simplification Bias** | Maintained full granular multiclass `OrderStatus` | Detected specific anomaly types without binary reduction |
| **Misleading Accuracy Metrics** | Evaluated via F1-Macro, Recall, and Multiclass ROC-AUC | Equitable performance assessment across all order statuses |

---

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=for-the-badge&logo=python&logoColor=white)

---

## 🔧 Pipeline Breakdown

Raw Data ➔ Feature Selection & Cleaning ➔ One-Hot Encoding & Date Parsing ➔ Stratified Train/Test Split
│
┌────────────────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────────────┐
▼                                                                                                                                         ▼
┌──────────────────────────────────────────────┐                                                                                         ┌──────────────────────────────────────────────┐
│       Logistic Regression Pipeline           │                                                                                         │            Random Forest Pipeline            │
│  1. StandardScaler                           │                                                                                         │  1. SMOTE (Multiclass Synthetic Sampling)    │
│  2. SMOTE (Multiclass Synthetic Sampling)    │                                                                                         │  2. RandomForestClassifier                   │
│  3. LogisticRegression(multi_class='ovr')    │                                                                                         │  3. Hyperparameter Tuning via GridSearchCV   │
│  4. Hyperparameter Tuning via GridSearchCV   │                                                                                         └──────────────────────┬───────────────────────┘
└──────────────────────┬───────────────────────┘                                                                                                                │
│                                                                                                                                        │
└───────────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┘
▼
Final Model Evaluation on Test Set
(F1-Macro, Multiclass ROC-AUC, Confusion Matrix)


### 1️⃣ Preprocessing & Feature Engineering
- Removed non-predictive tracking and identifier strings to prevent overfitting.
- Parsed date attributes into numerical temporal features.
- Applied One-Hot Encoding to convert categorical features into numerical vectors.

### 2️⃣ Resampling & Zero Data Leakage
- Used `imblearn.pipeline.Pipeline` to wrap **StandardScaler** and **SMOTE**.
- Executed synthetic oversampling strictly within individual cross-validation training folds to ensure test set leakage.

### 3️⃣ Model Exploration & Hyperparameter Tuning
- **Linear Baseline:** Evaluated `LogisticRegression` using a One-vs-Rest (`ovr`) strategy.
- **Non-Linear Ensemble:** Trained a `RandomForestClassifier` to capture complex non-linear feature interactions.
- Optimized both models via `GridSearchCV` targeting **F1-Macro** performance.

---

## 📊 Evaluation & Metrics Strategy

Standard overall accuracy was discarded in favor of robust evaluation metrics:
* **F1-Macro Score:** Measures the unweighted mean of F1-scores across all classes, ensuring rare fraud status types are given equal weight.
* **Recall:** Measures the proportion of actual fraudulent or risky transactions successfully caught by the model.
* **Multiclass ROC-AUC (Macro):** Evaluates the model's overall ability to discriminate between each specific class and all other classes combined.

---

## 💡 Key Learnings

- **Tree Ensembles vs. Linear Models:** Random Forest outperformed Logistic Regression due to its inherent ability to capture complex non-linear feature interactions without manual interaction engineering.
- **Pipeline Integrity:** Enforcing SMOTE resampling inside cross-validation folds via `imbalanced-learn` prevented synthetic data leakage, leading to realistic and reliable holdout evaluation.
- **Metric Alignment:** Relying on accuracy in severely imbalanced datasets creates a false sense of security; F1-Macro and Recall provide true performance visibility.

---

## 👩🏾‍💻 About Me

*Adeniyi Rukayat* — Data & Operations Analyst | Abuja, Nigeria

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/)
[![Fiverr](https://img.shields.io/badge/Fiverr-1DBF73?style=for-the-badge&logo=fiverr&logoColor=white)](https://www.fiverr.com/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:example@email.com)
[![Portfolio](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/databyrukayat)

*DecodeLabs Data Science Internship Program · 2026*
