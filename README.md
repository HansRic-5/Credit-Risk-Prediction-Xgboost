# Multi-Class Credit Score Classification Pipeline

An end-to-end Machine Learning pipeline designed to predict credit score ratings (`Good`, `Standard`, `Poor`) from heterogeneous customer financial data. This project handles real-world noisy tabular data through rigorous domain-specific preprocessing, loan frequency feature engineering, and robust pipeline modeling.

---

## 📌 Project Overview

Financial institutions rely on credit scoring to assess borrowing capacity and default risk. This repository implements a multi-class predictive system built on a credit score dataset containing 50,000 records with substantial data quality challenges, including corrupt strings, invalid numeric ranges, and mixed categorical attributes.

### Key Objectives
- Clean corrupted strings, formatting artifacts, and erroneous negative values across demographic and financial attributes.
- Transform multi-value compound attributes (`Type_of_Loan`) into structured frequency-based numerical signals.
- Standardize temporal variables into normalized numeric units (`Credit_History_Age` to months).
- Construct leakage-free preprocessing pipelines utilizing median/mode imputation, encoding, and tree-based ensembles (Random Forest & XGBoost).

---

## 🔍 Data Cleaning & Feature Engineering Highlights

1. **Identifier Elimination**: Dropped non-predictive metadata (`ID`, `Customer_ID`, `Name`, `SSN`).
2. **Noise and Symbol Stripping**:
   - Resolved formatting artifacts (trailing underscores `_`, placeholder symbols `_______`, `__10000__`, corrupted strings `!@9#%8`).
   - Standardized invalid categorical levels (e.g., converted placeholder `_` in `Credit_Mix` and `NM` in `Payment_of_Min_Amount` to `NaN` for imputation).
3. **Domain-Specific Sanity Filtering**:
   - Filtered applicant `Age` within a realistic demographic range (18 to 100 years).
   - Corrected negative values in strictly non-negative financial attributes (`Num_Bank_Accounts`, `Num_of_Loan`, `Delay_from_due_date`, `Num_of_Delayed_Payment`, `Monthly_Balance`) by thresholding at zero.
4. **Loan Frequency Decomposition**:
   - Disaggregated concatenated string values in `Type_of_Loan` into 8 discrete frequency indicators (`student_loan_freq`, `mortgage_loan_freq`, `payday_loan_freq`, etc.) to preserve debt exposure magnitude.
5. **Temporal Standardization**:
   - Converted string-formatted credit history (`"X Years and Y Months"`) into total numeric months.
6. **Target Encoding**:
   - Ordinally mapped `Credit_Score`: `Good` (2), `Standard` (1), `Poor` (0).

---

## 🛠️ Tech Stack & Modeling Pipeline

- **Languages & Core Libraries**: Python, NumPy, Pandas
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-Learn, Imbalanced-Learn, XGBoost
- **Pipeline Components**:
  - `ColumnTransformer` for separated numeric and categorical processing branches.
  - Imputation: `SimpleImputer(strategy='median')` for continuous metrics; `SimpleImputer(strategy='most_frequent')` for nominal attributes.
  - Resampling / Weighting: `SMOTE` / `compute_sample_weight` to address class imbalances.
  - Model Estimators: `RandomForestClassifier` and `XGBClassifier`.

---

## 📂 Repository Structure

```text
├── data/
│   └── Credis_Score_Dataset_B.csv   # Raw dataset (or link to data source)
├── notebooks/
│   └── credit_score_pipeline.ipynb  # Primary exploration, preprocessing & training
├── figures/
│   └── boxplots_outliers.png        # Generated EDA and outlier distribution plots
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation
