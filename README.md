# 🚀 Machine Learning Journey

Welcome to my Machine Learning Journey repository! Here, I document my progress, experiments, implementations, and projects as I learn machine learning from fundamental algorithms to advanced techniques.

---

## 📂 Repository Structure

```text
Machine-Learning-Journey/
│
├── .gitignore
├── README.md
├── requirements.txt
│
├── LinearRegression/
│   ├── insurance.csv            # Medical insurance cost dataset
│   └── code.ipynb               # Implementation of Linear Regression model
│
├── Houseprice Prediction/
│   ├── HousePricePrediction.csv # House pricing dataset
│   └── Code.ipynb               # EDA, preprocessing, and Linear Regression
│
├── LogisticRegression/
│   ├── 1-heart.csv              # Heart disease clinical dataset
│   └── Code.ipynb               # Binary classification using Logistic Regression
│
└── EmployeePrediction/
    ├── employee_turnover.csv    # Employee workplace & turnover dataset
    └── Code.ipynb               # Logistic Regression with L1 (Lasso) & L2 (Ridge) Regularization
```

---

## 📌 Projects & Topics Covered

### 1. Medical Insurance Cost Prediction (Linear Regression)
- **Folder:** [`LinearRegression/`](./LinearRegression/)
- **Notebook:** [`code.ipynb`](./LinearRegression/code.ipynb)
- **Dataset:** `insurance.csv` (1,338 records with columns: `age`, `sex`, `bmi`, `children`, `smoker`, `region`, `charges`)
- **Objective:** Predict individual medical charges billed by health insurance based on patient attributes.
- **Key Steps:**
  - Exploratory Data Analysis (EDA) using Seaborn scatter plots (`bmi` vs. `charges` colored by `smoker`).
  - Feature selection and categorical mapping (`sex`: female=1, male=0; `smoker`: yes=1, no=0).
  - Train-Test split (80% train, 20% test).
  - Model fitting with Scikit-Learn's `LinearRegression`.
  - Model evaluation using $R^2$ Score and Adjusted $R^2$.

### 2. House Price Prediction (Advanced Regression & EDA)
- **Folder:** [`Houseprice Prediction/`](./Houseprice%20Prediction/)
- **Notebook:** [`Code.ipynb`](./Houseprice%20Prediction/Code.ipynb)
- **Dataset:** `HousePricePrediction.csv` (2,919 records, features including lot area, building type, overall condition, year built, basement area, etc.)
- **Objective:** Predict residential house sale prices based on physical, zoning, and structural features.
- **Key Steps:**
  - In-depth Exploratory Data Analysis & descriptive statistics.
  - Missing value diagnosis and handling (`MSZoning`, `Exterior1st`, `BsmtFinSF2`, `TotalBsmtSF`).
  - Outlier detection using the Interquartile Range (IQR) method on `LotArea` and `SalePrice`.
  - One-hot encoding of categorical features with `pd.get_dummies`.
  - Train-Validation split (80% train, 20% validation).
  - Model fitting with Scikit-Learn's `LinearRegression`.
  - Model evaluation using $R^2$ Score (~0.62) and Adjusted $R^2$ (~0.56).
  - Scatter plot visualization of actual vs. predicted house prices with reference regression line.

### 3. Heart Disease Classification (Logistic Regression)
- **Folder:** [`LogisticRegression/`](./LogisticRegression/)
- **Notebook:** [`Code.ipynb`](./LogisticRegression/Code.ipynb)
- **Dataset:** `1-heart.csv` (303 records, 14 medical features: `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`, `target`)
- **Objective:** Predict the presence of heart disease (binary classification: 1 = disease, 0 = no disease) based on clinical health metrics.
- **Key Steps:**
  - Data inspection and missing value verification.
  - Feature-target separation (`X` features vs. `y` target) and class distribution inspection.
  - Train-Test split (80% train, 20% test, `random_state=42`).
  - Model fitting with Scikit-Learn's `LogisticRegression(max_iter=2000)`.
  - Comprehensive classification evaluation:
    - **Accuracy:** ~88.52%
    - **Precision:** ~87.88%
    - **Recall:** ~90.62%
    - **F1-Score:** ~89.23%
    - **Confusion Matrix:** 25 True Negatives, 29 True Positives, 4 False Positives, 3 False Negatives.

### 4. Employee Turnover Prediction (Regularized Logistic Regression)
- **Folder:** [`EmployeePrediction/`](./EmployeePrediction/)
- **Notebook:** [`Code.ipynb`](./EmployeePrediction/Code.ipynb)
- **Dataset:** `employee_turnover.csv` (1,350 records, 16 features including job satisfaction, monthly income, distance from home, work-life balance, and interaction terms)
- **Objective:** Predict whether an employee will leave the company (`Employee_Turnover`: 1 = leave, 0 = stay).
- **Key Steps & Regularization Comparison:**
  - Missing value audit and feature/target separation.
  - Train-Test split (80% train, 20% test, `random_state=42`).
  - Trained three variants of Logistic Regression:
    1. **Baseline**: Unregularized default Logistic Regression.
    2. **L1 Lasso (`C=0.5`, solver=`liblinear`)**: Encourages coefficient sparsity and feature selection.
    3. **L2 Ridge (`C=1.0`)**: Penalizes large coefficient weights to reduce overfitting.
  - Performance Results:
    - **Baseline Accuracy:** ~85.93% (F1: 0.84 on turnover class)
    - **L1 Lasso Accuracy:** **~87.04%** (F1: **0.86** on turnover class — top performer)
    - **L2 Ridge Accuracy:** ~85.93% (F1: 0.84 on turnover class)
  - Detailed classification reports comparing precision, recall, and F1 across all models.

---

## 🛠️ Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/Utkrish26/Machine-leaning-journey.git
   cd Machine-learning-journey
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install required libraries:
   ```bash
   pip install -r requirements.txt
   ```

4. Launch Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

---

## 📈 Learning Roadmap
- [x] Simple & Multiple Linear Regression
- [x] Advanced Linear Regression & Feature Engineering (House Price Prediction)
- [x] Logistic Regression & Binary Classification (Heart Disease Prediction)
- [x] Regularization Techniques: L1 Lasso & L2 Ridge (Employee Turnover Prediction)
- [ ] Decision Trees & Random Forests
- [ ] Support Vector Machines (SVM)
- [ ] K-Means Clustering
- [ ] Neural Networks & Deep Learning Basics
