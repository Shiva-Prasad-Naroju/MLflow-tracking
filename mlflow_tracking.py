import mlflow

# Set MLflow tracking URI (default is local storage)
mlflow.set_tracking_uri("file:./mlruns")

# List all experiments
experiments = mlflow.search_experiments()
for exp in experiments:
    print(f"Experiment: {exp.name}, ID: {exp.experiment_id}")

print("To view results, run: mlflow ui")
