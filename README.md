# Energy Efficiency ML Explorer

## Project Overview

This project is an interactive Streamlit application that explores how building design characteristics influence energy demand. Users can train and compare supervised machine learning models to predict building heating and cooling loads based on structural features.

The app is designed to demonstrate key machine learning concepts including model selection, hyperparameter tuning, and performance evaluation in a sustainability context.

---

## Live App

(https://energy-efficiency-ml-explorer.streamlit.app/)

---

## Features

- Upload your own dataset or use a built-in energy efficiency dataset
- Select a target variable (e.g., Heating Load or Cooling Load)
- Choose from multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
- Adjust hyperparameters using interactive controls
- Evaluate model performance using:
  - R² (coefficient of determination)
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
- Visualize results with:
  - Actual vs Predicted plots
  - Residual plots
  - Feature importance (for tree-based models)
- Compare model performance side-by-side
- Generate predictions from custom user input

---

## App Preview

### Model Performance Output

![Model Performance](https://raw.githubusercontent.com/Rbektas39/MLStreamlitApp/main/images/model_performance.png)

## How to Run Locally

1. Clone the repository:

```bash
git clone https://github.com/Rbektas39/MLStreamlitApp.git
cd MLStreamlitApp