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
└── LinearRegression/
    ├── insurance.csv       # Medical insurance cost dataset
    └── code.ipynb          # Implementation of Linear Regression model
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

---

## 🛠️ Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed.

### Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/<your-username>/machine-learning-journey.git
   cd machine-learning-journey
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
- [ ] Logistic Regression & Classification
- [ ] Decision Trees & Random Forests
- [ ] Support Vector Machines (SVM)
- [ ] K-Means Clustering
- [ ] Neural Networks & Deep Learning Basics
