import sys
import os
import yaml
import joblib
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score
from scipy.sparse import csr_matrix
from mlflow.tracking import MlflowClient

# from mlflow.exceptions import MlflowException

import pandas as pd


def train_model_rf(param_yaml_path):
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow_server:5000")
    )

    experiment_name = "Sentiment Analysis Experiment"
    mlflow.set_experiment(experiment_name)

    # Load parameters from YAML
    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)

    # Load datasets
    total_dataset_path = os.path.join(
        params_yaml["train"]["dir"], params_yaml["train"]["total_file"]
    )
    train_dataset_path = os.path.join(
        params_yaml["train"]["dir"], params_yaml["train"]["train_file"]
    )
    val_dataset_path = os.path.join(
        params_yaml["train"]["dir"], params_yaml["train"]["val_file"]
    )
    test_dataset_path = os.path.join(
        params_yaml["train"]["dir"], params_yaml["train"]["test_file"]
    )

    total_dataset_path = os.path.join(
        params_yaml["train"]["dir"], params_yaml["train"]["total_file"]
    )

    df_train = pd.read_csv(total_dataset_path)
    X_total, Y_total = df_train["feature"], df_train["label"]

    df_train = pd.read_csv(train_dataset_path)
    X_train, Y_train = df_train["feature"], df_train["label"]

    df_val = pd.read_csv(val_dataset_path)
    X_val, Y_val = df_val["feature"], df_val["label"]

    df_test = pd.read_csv(test_dataset_path)
    X_test, Y_test = df_test["feature"], df_test["label"]

    # Preprocessing
    cv = CountVectorizer(min_df=0.0, max_df=1.0, binary=False, ngram_range=(1, 3))
    X_total = cv.fit_transform(X_total)
    X_train = cv.transform(X_train)
    X_val = cv.transform(X_val)
    X_test = cv.transform(X_test)

    X_train = csr_matrix(X_train)
    X_val = csr_matrix(X_val)
    X_test = csr_matrix(X_test)

    lb = LabelBinarizer()
    Y_total = lb.fit_transform(Y_total).ravel()
    Y_train = lb.transform(Y_train).ravel()
    Y_val = lb.transform(Y_val).ravel()
    Y_test = lb.transform(Y_test).ravel()

    # Set MLflow tracking URI
    load_dotenv()

    with mlflow.start_run() as run:
        # Log parameters
        mlflow.log_param("loss", "hinge")
        mlflow.log_param("max_iter", 500)
        mlflow.log_param("random_state", 42)

        classifier = SGDClassifier(loss="modified_huber", max_iter=500, random_state=42)
        classifier.fit(X_train, Y_train)

        # Testing
        predicted_test = classifier.predict(X_test)
        accuracy = accuracy_score(Y_test, predicted_test)
        report = classification_report(
            Y_test,
            predicted_test,
            target_names=["Positive", "Negative"],
            output_dict=True,
        )

        print(report)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", report["weighted avg"]["precision"])
        mlflow.log_metric("recall", report["weighted avg"]["f1-score"])

        os.makedirs(
            os.path.join("/app", params_yaml["train"]["model_dir"]), exist_ok=True
        )

        # Define the model path
        model_path = os.path.join(
            "/app",  # Ensure the base path is correct within the container
            params_yaml["train"]["model_dir"],
            params_yaml["train"]["model_file"],
        )

        # Define the model path
        vectorizer_path = os.path.join(
            "/app",  # Ensure the base path is correct within the container
            params_yaml["train"]["model_dir"],
            params_yaml["train"]["vectorizer_file"],
        )

        joblib.dump(classifier, model_path)
        joblib.dump(cv, vectorizer_path)

        # Log artifacts
        # mlflow.log_artifact(model_path)
        mlflow.log_artifact(vectorizer_path)

        # Log model
        mlflow.sklearn.log_model(
            sk_model=classifier,
            artifact_path="model",
            registered_model_name="SentimentAnalysisModel",  # ,  # Register the model
        )

        # Register and transition to Production
        client = MlflowClient()
        model_name = "SentimentAnalysisModel"
        latest_version_info = client.get_latest_versions(model_name, stages=["None"])[0]
        client.transition_model_version_stage(
            name=model_name,
            version=latest_version_info.version,
            stage="Production",
            archive_existing_versions=True,  # ,
        )

        # print(f"Model saved at {model_path}")

        # Register the model
        run_id = run.info.run_id
        print(run_id)
        mlflow.register_model(f"runs:/{run_id}/model", "SentimentAnalysisModel")

        # End the run
        mlflow.end_run()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Usage: python train.py <path to params.yaml>")
        sys.exit(1)

    param_yaml_path = sys.argv[1]

    with open(param_yaml_path) as f:
        params_yaml = yaml.safe_load(f)

    train_model_rf(param_yaml_path)
