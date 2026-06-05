# churn_prediction
# Customer Churn Prediction & Hyperparameter Tuning using ANN

---

## Author
* **Tiayo Durel**

---

## Overview
This repository contains a deep learning project aimed at predicting bank customer churn using **Artificial Neural Networks (ANN)**. It includes a complete data preprocessing pipeline, baseline model training with real-time tracking, and an automated framework for optimizing network architecture via grid search hyperparameter tuning.

---

## Features
* **Data Cleansing**: Drops non-predictive tracking columns (`RowNumber`, `CustomerId`, `Surname`) to keep the model focused purely on behavioral and financial indicators.
* **Categorical Encoding**: Maps the binary `Gender` column using `LabelEncoder` and expands geographical entries (`France`, `Germany`, `Spain`) using `OneHotEncoder`.
* **Feature Scaling**: Implements a `StandardScaler` to handle magnitude variations among numeric variables like credit scores, balances, and estimated salaries.
* **Deep Learning Model**: Constructs a multi-layer feedforward neural network built with the `TensorFlow Keras Sequential API`.
* **Hyperparameter Optimization**: Employs `GridSearchCV` paired with `KerasClassifier` to dynamically evaluate optimal layer depths, neuron densities, and training epochs.
* **Training Callbacks**: Integrates `EarlyStopping` to halt training when validation loss ceases to improve alongside `TensorBoard` for tracking accuracy metrics dynamically.

---

## Project Structure
The main pipelines are divided into two primary operational steps based on the development workflows:
1. **Baseline ANN Model Training**: Performs complete data preprocessing, standardizes variables, builds a 3-layer architecture, evaluates performance, and saves the finalized artifacts.
2. **Hyperparameter Optimization**: Runs an automated grid search pipeline across various architectural complexities to identify optimal parameters.

---

## Installation & Environment Setup

To run the notebooks and scripts locally, set up your environment and install the required dependencies:

```bash
# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install necessary packages
pip install pandas scikit-learn tensorflow scikeras
