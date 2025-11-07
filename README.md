# Comparative Evaluation of Churn Prediction Using XGBoost, Random Forest, and EBM

## 📘 Project Overview
This project investigates customer churn prediction in the telecommunications industry using three supervised learning algorithms:
- **XGBoost**
- **Random Forest**
- **Explainable Boosting Machine (EBM)**

The goal is to balance **predictive accuracy**, **interpretability**, and **business relevance**.  
The notebook `MachineLearning1Ass2Coding.ipynb` contains all implementation stages, while the accompanying report (`MachineLearningAss-SangeethaRavichandran.pdf`) provides theoretical and analytical context.

---

## 📂 Notebook Structure

### 1. **Data Cleaning and Preprocessing**
- Checked and handled missing, duplicate, and outlier values.  
- Converted *TotalCharges* to numeric and replaced non-numeric values with the **median**.  
- Filled missing categorical values with `"Unknown"`.  
- Removed exact duplicate rows.  
- Applied **IQR capping** to control outliers in `Tenure`, `MonthlyCharges`, and `TotalCharges`.  
- Separated numeric and categorical features for correct processing.

### 2. **Exploratory Data Analysis (EDA)**
- Examined feature distributions and churn balance (≈73% non-churn vs 27% churn).  
- Visualized relationships between tenure, contract type, charges, and churn status.  
- Identified that month-to-month contracts and higher monthly charges strongly correlate with churn.  
- Observed a strong positive correlation between `MonthlyCharges` and `TotalCharges`.

### 3. **Feature Engineering and Transformation**
- Created domain-specific features such as:
  - `Charges_x_Tenure`
  - `AvgMonthlyChargeByTenure`
- Applied **StandardScaler** to numeric data and **One-Hot Encoding** to categorical features.  
- Resolved class imbalance through weighted models instead of oversampling.

### 4. **Model Development**
- Trained and compared:
  - **Random Forest** – reliable baseline model.  
  - **XGBoost** – high accuracy and strong performance on imbalanced data.  
  - **EBM** – interpretable, additive glass-box model.  
- Used stratified 80/20 train-test split and 5-fold cross-validation.  
- Evaluated models using:
  - Accuracy  
  - Precision, Recall, and F1-Score  
  - ROC-AUC  

### 5. **Interpretability and Counterfactual Analysis**
- Applied **SHAP** values to XGBoost and Random Forest for feature contribution insights.  
- Used EBM’s **additive interpretability** for transparent risk analysis.  
- Conducted **counterfactual “what-if” scenarios** (e.g., lowering monthly charges or extending contract length to reduce churn risk).

---

## 📊 Results Summary

| Model | Accuracy | AUC | Strength |
|:-------|:---------:|:---:|:----------|
| **Random Forest** | Moderate | 0.820 | Balanced baseline |
| **XGBoost** | High | 0.821 | Best recall & precision |
| **EBM** | Competitive | **0.839** | Highest interpretability |

**Key Findings**
- XGBoost achieved the highest recall — best at identifying churners.  
- EBM achieved the highest AUC and provided the clearest explanations.  
- Random Forest offered stable baseline performance.  
- Main churn drivers: **Contract Type**, **Tenure**, and **MonthlyCharges**.

---

## ⚙️ Environment and Dependencies
To run the notebook, install:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn interpret shap
