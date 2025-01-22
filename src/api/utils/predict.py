import os
import yaml
import joblib

from src.api.utils.preprocess_data import preprocess_text
from scipy.sparse import csr_matrix

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))


class PredictSentiment:
    def __init__(self, param_yaml_path):
        with open(param_yaml_path) as f:
            params_yaml = yaml.safe_load(f)

        self.vectorizer_path = os.path.join(
            root_path,
            params_yaml["train"]["model_dir"],
            params_yaml["train"]["vectorizer_file"],
        )

        # Define the model path
        self.model_path = os.path.join(
            root_path,  # Ensure the base path is correct within the container
            params_yaml["train"]["model_dir"],
            params_yaml["train"]["model_file"],
        )

        # Define the model path
        # vectorizer_path = os.path.join(
        #    "/app",  # Ensure the base path is correct within the container
        #    params_yaml["train"]["model_dir"],
        #    params_yaml["train"]["vectorizer_file"],
        # )

        self.model_uri = "models:/SentimentAnalysisModel/Production"
        self.model_uri = "models:/SentimentAnalysisModel/26"

    def preprocess(self, text_data):
        text_data = preprocess_text(text_data)
        # print(text_data)

        text_data_list = []
        text_data_list.append(text_data)

        return csr_matrix(self.vectorizer.transform(text_data_list))

    def predict(self, text):
       
        self.vectorizer = joblib.load(self.vectorizer_path)
        self.model = joblib.load(self.model_path)

        preprocessed_data = self.preprocess(text)
        # print(preprocessed_data)
        return self.model.predict(preprocessed_data)
