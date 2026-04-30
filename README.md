# Energy Efficiency ML Explorer

## Project Overview

The Energy Efficiency ML Explorer is an interactive Streamlit app that allows users to explore supervised machine learning models for predicting building energy demand. The app uses building design characteristics to predict either heating load or cooling load.

Users can work with the built-in energy efficiency dataset or upload their own CSV file. The app allows users to select predictor variables, choose a target variable, adjust model hyperparameters, compare models, and evaluate prediction performance using numerical metrics and visualizations.

## Live App

(https://energy-efficiency-ml-explorer.streamlit.app/)

## Repository

(https://github.com/Rbektas39/Bektas---Data---Science---Portfolio)

## Features

- Use a built-in energy efficiency dataset
- Upload a custom CSV dataset
- Select a target variable for prediction
- Select predictor variables
- Train and evaluate multiple regression models
- Adjust model hyperparameters
- Compare model performance across models
- View actual vs. predicted plots
- View residual plots
- View feature importance for tree-based models
- Make a prediction using custom input values

## Machine Learning Models

The app includes three supervised regression models:

1. **Linear Regression**
   - A baseline model that estimates a linear relationship between predictor variables and the target variable.

2. **Decision Tree Regressor**
   - A tree-based model that splits the data into decision rules. It can capture non-linear relationships but may overfit if the tree becomes too complex.

3. **Random Forest Regressor**
   - An ensemble model that combines multiple decision trees. It often performs better than a single decision tree because it averages across many trees.

## Data Requirements

Users can upload their own dataset in CSV format. However, the app currently uses only numeric columns for modeling.

Non-numeric columns, including categorical variables such as names, labels, regions, building types, or categories, are automatically dropped during preprocessing. If users want categorical variables included in the model, they need to encode those variables before uploading the dataset.

For example, a categorical column like:

```text
Building Type: Office, School, Residential
