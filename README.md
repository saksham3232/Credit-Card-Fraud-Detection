# Credit Card Fraud Detection

This repository contains a project focused on detecting fraudulent credit card transactions using machine learning techniques. The dataset used for this project comes from anonymized transactions and is widely used for fraud detection research.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Credit card fraud is a significant issue for financial institutions and their customers. This project aims to build a machine learning model that can accurately classify transactions as fraudulent or legitimate using a dataset of anonymized transactions.

## Dataset

The dataset used in this project is the [Credit Card Fraud Detection Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud). It contains transactions made by European cardholders over two days in September 2013.

- **Number of Transactions**: 284,807
- **Number of Fraudulent Transactions**: 492
- **Features**: 30 (including anonymized features `V1, V2, ..., V28`, `Time`, and `Amount`)
- **Class Distribution**: Highly imbalanced (fraudulent transactions constitute only ~0.17% of the dataset)

## Project Workflow

The project workflow includes the following steps:

1. **Data Preprocessing**:
   - Handling missing values
   - Scaling features (e.g., `Amount`, `Time`)
   - Balancing the dataset using techniques like oversampling (SMOTE) or undersampling

2. **Exploratory Data Analysis (EDA)**:
   - Visualizing the class distribution
   - Analyzing feature correlations

3. **Model Training and Evaluation**:
   - Training machine learning models like Logistic Regression, Random Forest, Gradient Boosting, etc.
   - Evaluating model performance using metrics like accuracy, precision, recall, F1-score, and ROC-AUC

4. **Deployment**:
   - Saving the trained model for future use

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/saksham3232/Credit-Card-Fraud-Detection.git
   cd Credit-Card-Fraud-Detection
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Open the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```

## Usage

1. Open the `Credit Card Fraud Detection.ipynb` notebook.
2. Follow the steps outlined in the notebook to preprocess the data, train models, and evaluate their performance.
3. Modify the notebook as needed to experiment with different models or preprocessing techniques.

## Results

The project achieves significant improvement in fraud detection through the use of advanced machine learning techniques. The key results include:

- Improved recall for fraudulent transactions
- Balanced precision-recall tradeoff

For detailed results, refer to the output sections of the Jupyter Notebook.

## Technologies Used

- **Programming Language**: Python
- **Libraries**:
  - Jupyter Notebook
  - Pandas, NumPy for data preprocessing
  - Scikit-learn for machine learning
  - Matplotlib, Seaborn for data visualization
  - Imbalanced-learn for handling imbalanced datasets

## Contributing

Contributions are welcome! If you have ideas to improve the project or want to fix bugs, feel free to fork the repository, make your changes, and submit a pull request.

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature-name`)
3. Commit your changes (`git commit -m 'Add feature'`)
4. Push to the branch (`git push origin feature-name`)
5. Open a pull request
