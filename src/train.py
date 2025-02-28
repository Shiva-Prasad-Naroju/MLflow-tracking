import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
data = pd.read_csv("C:\\Users\\DELL\\Downloads\\GitHub\\MLflow-tracking\\data\\sample_data.csv")
X = data[['feature']]
y = data['target']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow experiment
mlflow.set_experiment("Linear_Regression_Experiment")

with mlflow.start_run():
    # Model training
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)

    # Log parameters and metrics
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_metric("mse", mse)

    # Log model
    mlflow.sklearn.log_model(model, "linear_regression_model")

    print(f"Model trained and logged with MSE: {mse}")
