
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


st.set_page_config(
    page_title="California Housing ML",
    layout="wide"
)

st.title("California Housing Price Prediction")
st.write("Linear Regression, Lasso and Ridge Regression")


@st.cache_data
def load_dataset():

    data = fetch_california_housing(as_frame=True)

    df = data.frame.copy()

    df["RoomsPerPerson"] = df["AveRooms"] / (df["Population"] + 1)
    df["BedroomsPerRoom"] = df["AveBedrms"] / (df["AveRooms"] + 1)
    df["PopulationPerHousehold"] = df["Population"] / (df["AveOccup"] + 1)
    df["RoomsPerHousehold"] = df["AveRooms"] / (df["AveOccup"] + 1)
    df["IncomePerRoom"] = df["MedInc"] / (df["AveRooms"] + 1)

    return df


df = load_dataset()

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Linear Regression

linear_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

linear_model.fit(X_train, y_train)

linear_pred = linear_model.predict(X_test)


# Lasso

lasso_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Lasso(max_iter=10000))
])

lasso_grid = GridSearchCV(
    lasso_model,
    {
        "model__alpha": np.logspace(-4, 1, 20)
    },
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

lasso_grid.fit(X_train, y_train)

lasso_pred = lasso_grid.predict(X_test)


# Ridge

ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

ridge_grid = GridSearchCV(
    ridge_model,
    {
        "model__alpha": np.logspace(-4, 4, 20)
    },
    cv=5,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

ridge_grid.fit(X_train, y_train)

ridge_pred = ridge_grid.predict(X_test)


# Metrics

def get_metrics(actual, predicted):

    mae = mean_absolute_error(actual, predicted)

    mse = mean_squared_error(actual, predicted)

    rmse = np.sqrt(mse)

    r2 = r2_score(actual, predicted)

    return mae, mse, rmse, r2


linear_metrics = get_metrics(y_test, linear_pred)
lasso_metrics = get_metrics(y_test, lasso_pred)
ridge_metrics = get_metrics(y_test, ridge_pred)


# Sidebar

st.sidebar.title("Menu")

option = st.sidebar.selectbox(
    "Choose Section",
    [
        "Dataset",
        "Model Performance",
        "Prediction",
        "Coefficients",
        "Alpha Analysis"
    ]
)


# Dataset

if option == "Dataset":

    st.header("Dataset")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])

    col2.metric("Features", X.shape[1])

    col3.metric("Target", "MedHouseVal")

    st.subheader("Dataset Preview")

    st.dataframe(df.head(10))

    st.subheader("Statistics")

    st.dataframe(df.describe())


# Model Performance

elif option == "Model Performance":

    st.header("Model Performance")

    results = pd.DataFrame({
        "Model": [
            "Linear Regression",
            "Lasso Regression",
            "Ridge Regression"
        ],
        "MAE": [
            linear_metrics[0],
            lasso_metrics[0],
            ridge_metrics[0]
        ],
        "MSE": [
            linear_metrics[1],
            lasso_metrics[1],
            ridge_metrics[1]
        ],
        "RMSE": [
            linear_metrics[2],
            lasso_metrics[2],
            ridge_metrics[2]
        ],
        "R2": [
            linear_metrics[3],
            lasso_metrics[3],
            ridge_metrics[3]
        ]
    })

    st.dataframe(results.round(4))

    st.subheader("Error Comparison")

    fig, ax = plt.subplots()

    results.set_index("Model")[["MAE", "RMSE"]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Error")

    st.pyplot(fig)

    st.subheader("R2 Score")

    fig2, ax2 = plt.subplots()

    results.set_index("Model")["R2"].plot(
        kind="bar",
        ax=ax2
    )

    ax2.set_ylabel("R2 Score")

    st.pyplot(fig2)

    st.write(
        "Best Lasso Alpha:",
        lasso_grid.best_params_["model__alpha"]
    )

    st.write(
        "Best Ridge Alpha:",
        ridge_grid.best_params_["model__alpha"]
    )


# Prediction

elif option == "Prediction":

    st.header("House Value Prediction")

    st.write("Enter the required values.")

    values = {}

    for feature in X.columns:

        values[feature] = st.number_input(
            feature,
            value=float(X[feature].median())
        )

    input_data = pd.DataFrame([values])

    model_name = st.selectbox(
        "Select Model",
        [
            "Linear Regression",
            "Lasso Regression",
            "Ridge Regression"
        ]
    )

    if st.button("Predict"):

        if model_name == "Linear Regression":

            prediction = linear_model.predict(
                input_data
            )[0]

        elif model_name == "Lasso Regression":

            prediction = lasso_grid.predict(
                input_data
            )[0]

        else:

            prediction = ridge_grid.predict(
                input_data
            )[0]

        st.success(
            f"Predicted House Value: ${prediction * 100000:,.2f}"
        )


# Coefficients

elif option == "Coefficients":

    st.header("Feature Coefficients")

    linear_coef = linear_model.named_steps["model"].coef_

    lasso_coef = lasso_grid.best_estimator_.named_steps["model"].coef_

    ridge_coef = ridge_grid.best_estimator_.named_steps["model"].coef_

    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Linear": linear_coef,
        "Lasso": lasso_coef,
        "Ridge": ridge_coef
    })

    st.dataframe(coef_df.round(4))

    fig, ax = plt.subplots(figsize=(12, 6))

    coef_df.set_index("Feature").plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Coefficient")

    ax.set_title("Coefficient Comparison")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)

    zero_features = coef_df[
        coef_df["Lasso"] == 0
    ]["Feature"].tolist()

    st.subheader("Lasso Zero Coefficients")

    if len(zero_features) > 0:

        for feature in zero_features:
            st.write("•", feature)

    else:

        st.write("No zero coefficients.")


# Alpha Analysis

elif option == "Alpha Analysis":

    st.header("Grid Search Analysis")

    lasso_results = pd.DataFrame(
        lasso_grid.cv_results_
    )

    lasso_alpha = lasso_results[
        "param_model__alpha"
    ].astype(float)

    lasso_error = -lasso_results[
        "mean_test_score"
    ]

    st.subheader("Lasso Alpha vs CV Error")

    fig, ax = plt.subplots()

    ax.semilogx(
        lasso_alpha,
        lasso_error,
        marker="o"
    )

    ax.set_xlabel("Alpha")

    ax.set_ylabel("CV MSE")

    st.pyplot(fig)


    ridge_results = pd.DataFrame(
        ridge_grid.cv_results_
    )

    ridge_alpha = ridge_results[
        "param_model__alpha"
    ].astype(float)

    ridge_error = -ridge_results[
        "mean_test_score"
    ]

    st.subheader("Ridge Alpha vs CV Error")

    fig2, ax2 = plt.subplots()

    ax2.semilogx(
        ridge_alpha,
        ridge_error,
        marker="o"
    )

    ax2.set_xlabel("Alpha")

    ax2.set_ylabel("CV MSE")

    st.pyplot(fig2)

    st.info(
        "Best Lasso Alpha: "
        + str(lasso_grid.best_params_["model__alpha"])
    )

    st.info(
        "Best Ridge Alpha: "
        + str(ridge_grid.best_params_["model__alpha"])
    )
