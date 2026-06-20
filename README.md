<div align="center">

# 📈 Sales Prediction using LightGBM & Optuna

### Advanced Retail Sales Forecasting with Machine Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()
[![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-green.svg)]()
[![Optuna](https://img.shields.io/badge/Optuna-Hyperparameter%20Optimization-orange.svg)]()
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-red.svg)]()

Predicting Walmart sales using advanced feature engineering, gradient boosting, and automated hyperparameter optimization.

</div>

---

## 🚀 Overview

This project implements an end-to-end sales forecasting pipeline using **LightGBM** and **Optuna** to predict retail sales on Walmart sales data.

The solution focuses on:

- 📊 Accurate sales forecasting
- ⚡ Efficient gradient boosting with LightGBM
- 🎯 Automated hyperparameter tuning using Optuna
- 🔍 Advanced feature engineering
- 📈 Model performance optimization
- 🏪 Retail sales trend analysis

The model captures temporal sales patterns, promotional impacts, seasonal trends, and store-level behavior to improve forecasting accuracy.

---

## ✨ Features

### 📌 Data Processing
- Data cleaning and preprocessing
- Missing value handling
- Feature encoding
- Outlier treatment

### 📌 Feature Engineering
- Date-based features
- Week, month, quarter extraction
- Lag features
- Rolling statistics
- Seasonal indicators
- Holiday impact features

### 📌 Model Development
- LightGBM Regressor
- Cross-validation strategy
- Feature importance analysis
- Performance monitoring

### 📌 Hyperparameter Optimization
- Optuna-based tuning
- Automated parameter search
- Efficient trial management
- Best model selection

### 📌 Evaluation
- RMSE
- MAE
- R² Score
- Prediction visualization

---

## 🏗️ Project Architecture

```text
Raw Walmart Data
        │
        ▼
Data Preprocessing
        │
        ▼
Feature Engineering
        │
        ▼
Train/Test Split
        │
        ▼
Optuna Optimization
        │
        ▼
LightGBM Training
        │
        ▼
Model Evaluation
        │
        ▼
Sales Forecasting
```

---

## 🛠️ Tech Stack

| Category | Technology |
|-----------|------------|
| Language | Python |
| ML Model | LightGBM |
| Optimization | Optuna |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Evaluation | Scikit-Learn |
| Notebook Environment | Jupyter Notebook |

---

## 📂 Project Structure

```text
sales_prediction/
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── features.csv
│
├── notebooks/
│   └── sales_prediction.ipynb
│
├── models/
│   └── trained_model.pkl
│
├── images/
│   └── results.png
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/omkaravasare1/sales_prediction.git
cd sales_prediction
```

### Create Virtual Environment

```bash
python -m venv venv
```

Activate:

```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

```bash
jupyter notebook
```

Open the notebook and execute all cells.

Or run your training script:

```bash
python train.py
```

---

## 📊 Model Pipeline

### 1️⃣ Data Preparation
- Clean raw sales data
- Handle missing values
- Merge store and feature datasets

### 2️⃣ Feature Engineering
- Extract date components
- Create lag variables
- Generate rolling averages
- Build seasonal indicators

### 3️⃣ Hyperparameter Optimization
Optuna searches for the best:

- Learning Rate
- Number of Leaves
- Max Depth
- Feature Fraction
- Bagging Fraction
- Minimum Child Samples

### 4️⃣ Model Training
LightGBM learns complex relationships between:

- Store performance
- Seasonal demand
- Promotions
- Holidays
- Historical sales

### 5️⃣ Evaluation
Performance is measured using:

```text
RMSE (Root Mean Squared Error)
MAE  (Mean Absolute Error)
R²   (Coefficient of Determination)
```

---

## 📈 Results

The optimized LightGBM model successfully captures:

✅ Seasonal sales fluctuations

✅ Store-level purchasing patterns

✅ Holiday demand spikes

✅ Historical trend dependencies

✅ Improved forecasting accuracy through Optuna tuning

---

## 🔥 Why LightGBM?

LightGBM was selected because it:

- Trains significantly faster than traditional boosting models
- Handles large datasets efficiently
- Provides strong predictive performance
- Supports advanced feature importance analysis
- Works exceptionally well for tabular business data

---

## 📸 Sample Workflow

```text
Walmart Dataset
      ↓
Feature Engineering
      ↓
Optuna Optimization
      ↓
LightGBM Training
      ↓
Model Evaluation
      ↓
Future Sales Prediction
```

---

## 🔮 Future Improvements

- Deep Learning (LSTM / Transformer Models)
- Real-time Forecasting Dashboard
- Model Deployment using FastAPI
- Interactive Streamlit Application
- Automated Data Pipeline
- Explainable AI using SHAP

---

## 🤝 Contributing

Contributions, suggestions, and improvements are welcome.

1. Fork the repository
2. Create a feature branch

```bash
git checkout -b feature/new-feature
```

3. Commit changes

```bash
git commit -m "Added new feature"
```

4. Push

```bash
git push origin feature/new-feature
```

5. Open a Pull Request

---

## 👨‍💻 Author

### Omkar Avasare

Third-Year Information Technology Student | Machine Learning Enthusiast | Cloud & FinTech Explorer

- 💼 Interested in AI/ML, FinTech, Cloud Computing
- 📈 Passionate about Data Science and Predictive Analytics

GitHub: https://github.com/omkaravasare1

---

<div align="center">

### ⭐ If you found this project useful, consider giving it a star!

**Built with Python, LightGBM, and Optuna**

</div>
