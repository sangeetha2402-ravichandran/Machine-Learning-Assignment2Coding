# Machine Learning Assignment 2 – Retail Sales Prediction

## 📘 Project Overview
This project explores predictive modeling on a retail sales dataset using supervised learning algorithms.  
The aim is to identify key factors affecting product sales and to develop regression models that can accurately estimate the sales based on product and outlet attributes.

The notebook `MachineLearning1Ass2Coding.ipynb` contains all stages of the workflow — from data cleaning and visualization to model training and performance evaluation.

---

## 📂 Main Sections of the Notebook

### 1. **Data Cleaning and Preprocessing**
- Loaded and inspected the raw dataset.
- Removed exact duplicate rows to ensure unique observations.
- Converted numerical and categorical data types appropriately.
- Treated missing values:
  - Numeric columns filled with median values.
  - Categorical columns imputed with mode.
- Identified and removed ID-like columns (e.g., `Item_Identifier`).
- Applied IQR capping to handle outliers in `Item_Weight`, `Item_Visibility`, and `Item_MRP`.

### 2. **Exploratory Data Analysis (EDA)**
- Univariate, bivariate, and multivariate analysis of key features.
- Visualized data distributions using histograms, bar charts, and violin plots.
- Explored relationships between product type, visibility, MRP, outlet type, and sales.
- Identified key insights such as:
  - Higher MRP items typically have higher sales.
  - Outlet size and establishment year influence overall sales.
  - Certain product categories underperform consistently.

### 3. **Feature Engineering**
- Created new derived features such as:
  - `Price_per_Weight`
  - `Visibility_Adjusted`
  - `Category_Sales_Avg`
- Encoded categorical features using Label Encoding and One-Hot Encoding.
- Normalized or scaled numerical features where applicable.

### 4. **Model Development**
- Implemented and compared multiple regression models:
  - Linear Regression  
  - Ridge and Lasso Regression  
  - ElasticNet  
  - Random Forest Regressor  
- Split dataset into training and testing subsets.
- Evaluated model performance using:
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - Mean Absolute Error (MAE)  
  - R² Score

### 5. **Results and Evaluation**
- Compared performance across all models.
- Random Forest achieved the highest accuracy and lowest error.
- Ridge and Lasso models helped identify key regularized coefficients.
- Model interpretability explored through feature importance ranking.

---

## 📊 Result Summary

| Model | RMSE | MAE | R² Score |
|--------|------|------|-----------|
| Linear Regression | Moderate | Moderate | Low |
| Ridge Regression | Improved | Improved | Better stability |
| Lasso Regression | Slightly reduced error | Good generalization | Moderate |
| Random Forest | **Lowest error** | **Highest accuracy** | **Best overall** |

**Conclusion:**  
Random Forest outperformed other models for retail sales prediction due to its ability to capture nonlinear patterns and feature interactions.  
Further improvement could be achieved through hyperparameter tuning and inclusion of external market factors.

---

## ⚙️ Environment & Dependencies
To run the notebook, install the following Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
