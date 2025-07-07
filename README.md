# Credit Card Fraud Detection using Random Forest

This project is a complete pipeline for detecting fraudulent credit card transactions using machine learning. It covers data exploration, preprocessing (including class imbalance handling), model building, evaluation, and deployment as a web application using Flask.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Data](#data)
- [How it Works](#how-it-works)
- [Model Training Pipeline](#model-training-pipeline)
- [Web Application](#web-application)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

Credit card fraud is a serious problem for financial organizations and individuals. In this project, we use a Random Forest classifier to differentiate between legitimate and fraudulent transactions based on various anonymized features from a real-world dataset.

The repository contains:
- An end-to-end Jupyter notebook for data analysis, feature engineering, model training, and evaluation.
- A Flask web app for real-time prediction using the trained model.

---

## Features

- **Data Exploration & Visualization**: Understand feature distributions, class imbalance, and correlations.
- **Preprocessing**: Handles missing values, scaling, and severe class imbalance using SMOTE.
- **Model Building**: Random Forest classifier with hyperparameter tuning.
- **Evaluation**: Multiple metrics—accuracy, precision, recall, F1-score, confusion matrix, and more.
- **Deployment**: User-friendly Flask web app for making predictions on new data.

---

## Data

- The dataset used is [Kaggle's Credit Card Fraud Detection dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud).
- All features except `Time`, `Amount`, and `Class` are PCA-transformed for privacy.
- The dataset is highly imbalanced, with a very small fraction of fraudulent transactions.

---

## How it Works

1. **Exploration**: The notebook explores the dataset, checks class distributions, and visualizes correlations.
2. **Preprocessing**:
   - Checks for missing data.
   - Scales features using `StandardScaler`.
   - Handles class imbalance with `SMOTE` oversampling.
3. **Model Training**: 
   - Splits data into train/test sets.
   - Applies Random Forest with class weighting and optional hyperparameter tuning.
   - Evaluates on multiple metrics.
4. **Saving Artifacts**: 
   - Saves the trained scaler and model as `.pkl` files for inference.
5. **Web App**:
   - Flask app loads the scaler and model.
   - Accepts 30-feature input via web form.
   - Outputs whether the transaction is **Fraudulent** or **Legitimate**.

---

## Model Training Pipeline

All steps are documented in [`Credit Card Fraud Detection.ipynb`](Credit%20Card%20Fraud%20Detection.ipynb):

- Data loading
- Data inspection & preprocessing
- Correlation analysis
- Splitting features (`X`) and target (`y`)
- Feature scaling
- Train-test split (stratified)
- SMOTE for balancing
- Model training with `RandomForestClassifier`
- Model evaluation (accuracy, precision, recall, F1-score, confusion matrix)
- Saving model and scaler with `pickle`

---

## Web Application

The Flask app (`app.py`):

- **Loads** the trained scaler and model from the `models/` directory.
- **Homepage**: `index.html` (should be placed in the `templates/` folder).
- **Prediction**: Accepts comma-separated 30 features, returns prediction.
- **Error Handling**: Ensures correct input format.

---

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/saksham3232/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2. Install requirements

It is recommended to use a virtual environment.

```bash
pip install -r requirements.txt
```

**Typical requirements:**
- numpy
- pandas
- scikit-learn
- imbalanced-learn
- seaborn
- matplotlib
- flask

### 3. Train the model

Open and run through the notebook:

```bash
jupyter notebook "Credit Card Fraud Detection.ipynb"
```

This will generate `scaler.pkl` and `model.pkl` in the `models/` folder.

### 4. Run the web app

```bash
python app.py
```

Visit `http://127.0.0.1:5000/` in your browser.

### 5. Using the app

- Enter 30 comma-separated features in the form.
- Click **Predict**.
- The app will tell you if the transaction is fraudulent or legitimate.

---

## Project Structure

```
Credit-Card-Fraud-Detection/
├── app.py
├── Credit Card Fraud Detection.ipynb
├── models/
│   ├── model.pkl
│   └── scaler.pkl
├── templates/
│   └── index.html
├── requirements.txt
└── README.md
```

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests.

---

**Author:** [saksham3232](https://github.com/saksham3232)
