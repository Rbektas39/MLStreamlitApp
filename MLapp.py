import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


st.set_page_config(page_title="Energy Efficiency ML Explorer", layout="wide")

st.title("Energy Efficiency ML Explorer")

st.write(
    "This interactive app explores how building design characteristics affect energy demand. "
    "Users can train and compare supervised machine learning models, adjust hyperparameters, "
    "and evaluate performance when predicting heating or cooling load."
)

with st.expander("How this app works"):
    st.write("""
    - Use the sample energy dataset or upload your own CSV file.
    - The app uses only numeric columns for modeling.
    - Non-numeric columns are removed because the models require numeric inputs.
    - Select a target variable and predictor variables.
    - Choose a regression model and adjust its hyperparameters.
    - Evaluate model performance using R², MAE, RMSE, and visual plots.
    """)


@st.cache_data
def load_sample_data():
    df = pd.read_csv("data/energy_efficiency.csv")
    return df


def prepare_uploaded_data(df):
    """
    Prepares an uploaded dataset for modeling.

    The function currently returns a copy of the uploaded dataset.
    Later in the app, only numeric columns are selected for modeling.
    Non-numeric columns, including categorical variables, are not used unless
    the user encodes them before uploading the file.
    """
    return df.copy()


def get_model(model_name, params):
    if model_name == "Linear Regression":
        model = LinearRegression()

    elif model_name == "Decision Tree Regressor":
        model = DecisionTreeRegressor(
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=params["random_state"]
        )

    elif model_name == "Random Forest Regressor":
        model = RandomForestRegressor(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            random_state=params["random_state"]
        )

    else:
        model = None

    return model


def evaluate_model(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return r2, mae, rmse


def plot_actual_vs_predicted(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.7)
    ax.axhline(y=0, linestyle="--")
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals")
    ax.set_title("Residual Plot")
    st.pyplot(fig)


def plot_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(importance_df["Feature"], importance_df["Importance"])
        ax.set_title("Feature Importance")
        ax.invert_yaxis()
        st.pyplot(fig)


st.sidebar.header("App Controls")

data_option = st.sidebar.radio(
    "Choose a dataset source:",
    ["Use sample energy dataset", "Upload my own CSV"]
)

if data_option == "Use sample energy dataset":
    df = load_sample_data()

else:
    uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = prepare_uploaded_data(df)

        st.warning(
            "This app only uses numeric columns for modeling. Non-numeric columns, including categorical variables, "
            "are automatically dropped. If you want categorical variables included, please encode them before uploading your dataset."
        )

    else:
        st.warning("Please upload a CSV file to continue.")
        st.stop()


st.subheader("Dataset Preview")
st.dataframe(df.head())

numeric_df = df.select_dtypes(include=np.number)
numeric_columns = numeric_df.columns.tolist()

dropped_cols = set(df.columns) - set(numeric_columns)

if dropped_cols:
    st.info(f"Dropped non-numeric columns: {', '.join(dropped_cols)}")

if len(numeric_columns) < 2:
    st.error("Your dataset must contain at least two numeric columns.")
    st.stop()


default_target = None

if "Heating Load" in numeric_columns:
    default_target = "Heating Load"
elif "Cooling Load" in numeric_columns:
    default_target = "Cooling Load"
else:
    default_target = numeric_columns[-1]


target = st.selectbox(
    "Select the target variable:",
    options=numeric_columns,
    index=numeric_columns.index(default_target)
)

feature_options = [col for col in numeric_columns if col != target]

selected_features = st.multiselect(
    "Select predictor variables:",
    options=feature_options,
    default=feature_options
)

if len(selected_features) == 0:
    st.error("Please select at least one predictor variable.")
    st.stop()


model_name = st.sidebar.selectbox(
    "Select a regression model:",
    ["Linear Regression", "Decision Tree Regressor", "Random Forest Regressor"]
)

test_size = st.sidebar.slider(
    "Test set size",
    min_value=0.1,
    max_value=0.5,
    value=0.2,
    step=0.05
)

st.sidebar.caption(
    "Test size controls how much data is held out for testing. "
    "A larger test size leaves less data for model training."
)

random_state = st.sidebar.number_input(
    "Random state",
    min_value=0,
    max_value=1000,
    value=42,
    step=1
)

st.sidebar.caption(
    "Random state controls reproducibility. Using the same value gives the same train/test split each run."
)

params = {"random_state": random_state}

if model_name == "Decision Tree Regressor":
    params["max_depth"] = st.sidebar.slider("Max depth", min_value=1, max_value=30, value=5)
    st.sidebar.caption("Max depth controls how complex the tree can grow.")

    params["min_samples_split"] = st.sidebar.slider("Min samples split", min_value=2, max_value=20, value=2)
    st.sidebar.caption("Min samples split determines how many samples are needed before a node can be split.")

    params["min_samples_leaf"] = st.sidebar.slider("Min samples leaf", min_value=1, max_value=20, value=1)
    st.sidebar.caption("Min samples leaf sets the minimum number of samples required in a final leaf node.")

elif model_name == "Random Forest Regressor":
    params["n_estimators"] = st.sidebar.slider("Number of trees", min_value=10, max_value=300, value=100, step=10)
    st.sidebar.caption("Number of trees controls how many decision trees are combined in the forest.")

    params["max_depth"] = st.sidebar.slider("Max depth", min_value=1, max_value=30, value=8)
    st.sidebar.caption("Max depth limits how complex each individual tree can be.")

    params["min_samples_split"] = st.sidebar.slider("Min samples split", min_value=2, max_value=20, value=2)
    st.sidebar.caption("Min samples split controls how many samples are needed before a node can be split.")


X = numeric_df[selected_features]
y = numeric_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state
)

model = get_model(model_name, params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2, mae, rmse = evaluate_model(y_test, y_pred)


st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)
col1.metric("R²", f"{r2:.3f}")
col2.metric("MAE", f"{mae:.3f}")
col3.metric("RMSE", f"{rmse:.3f}")

st.caption(
    "R² shows how much variation in the target is explained by the model. "
    "MAE and RMSE measure prediction error, with lower values indicating better performance."
)

plot_col1, plot_col2 = st.columns(2)

with plot_col1:
    plot_actual_vs_predicted(y_test, y_pred)

with plot_col2:
    plot_residuals(y_test, y_pred)

if model_name in ["Decision Tree Regressor", "Random Forest Regressor"]:
    st.subheader("Feature Importance")
    plot_feature_importance(model, selected_features)


st.subheader("Compare All Models")

st.write(
    "This table compares all three regression models on the same training and test split."
)

if st.button("Compare all models"):
    comparison_results = []

    model_specs = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(
            random_state=random_state,
            max_depth=5
        ),
        "Random Forest Regressor": RandomForestRegressor(
            random_state=random_state,
            n_estimators=100,
            max_depth=8
        )
    }

    for name, test_model in model_specs.items():
        test_model.fit(X_train, y_train)
        preds = test_model.predict(X_test)
        test_r2, test_mae, test_rmse = evaluate_model(y_test, preds)

        comparison_results.append({
            "Model": name,
            "R²": round(test_r2, 3),
            "MAE": round(test_mae, 3),
            "RMSE": round(test_rmse, 3)
        })

    comparison_df = pd.DataFrame(comparison_results)
    st.dataframe(comparison_df)


st.subheader("Make a Prediction")

with st.form("prediction_form"):
    user_inputs = {}

    for feature in selected_features:
        min_val = float(numeric_df[feature].min())
        max_val = float(numeric_df[feature].max())
        mean_val = float(numeric_df[feature].mean())

        user_inputs[feature] = st.number_input(
            f"{feature}",
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted {target}: {prediction:.3f}")
