import os
import sys
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

from string import punctuation
from dotenv import load_dotenv

nltk.download("stopwords")


# Removing the html strips
def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup.get_text()


# Removing the square brackets
def remove_between_square_brackets(text):
    return re.sub(r"\[[^]]*\]", "", text)


# Removing the noisy text
def denoise_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def preprocess_data(df):
    stop = set(stopwords.words("english"))

    stop.discard("not")

    df["review"] = df["review"].apply(lambda x: x.lower())
    df["review"] = df["review"].apply(lambda x: remove_between_square_brackets(x))
    df["review"] = df["review"].apply(lambda x: denoise_text(x))

    df["review"] = df["review"].apply(
        lambda x: "".join([c for c in x if c not in punctuation])
    )
    df["review"] = df["review"].apply(
        lambda x: " ".join([c for c in x.split() if c not in stop])
    )

    return df


def preprocess_text(text):
    stop = set(stopwords.words("english"))
    stop.discard("not")

    text = text.lower()
    text = remove_between_square_brackets(text)
    text = denoise_text(text)
    text = "".join([c for c in text if c not in punctuation])
    text = " ".join([c for c in text.split() if c not in stop])

    return text


def split_data(features, encoded_labels, trim_step=1, split_frac=0.8, random_state=42):
    np.random.seed(random_state)

    len_feat = len(features)
    # Number of rows in the sparse matrix

    # Shuffle indices
    indices = np.arange(len_feat)
    np.random.shuffle(indices)

    # Reorder the data based on shuffled indices
    features = np.array(features)[indices]
    encoded_labels = np.array(encoded_labels)[indices]

    # Trim the data
    features = features[::trim_step]
    encoded_labels = encoded_labels[::trim_step]

    len_feat = features.shape[0]
    # Update length after trimming

    total_x = features
    total_y = encoded_labels

    # Split data
    split_index = int(split_frac * len_feat)
    train_x = features[:split_index]
    train_y = encoded_labels[:split_index]

    remaining_x = features[split_index:]
    remaining_y = encoded_labels[split_index:]

    valid_index = int(remaining_x.shape[0] * 0.5)
    valid_x = remaining_x[:valid_index]
    valid_y = remaining_y[:valid_index]

    test_x = remaining_x[valid_index:]
    test_y = remaining_y[valid_index:]

    return total_x, total_y, train_x, train_y, valid_x, valid_y, test_x, test_y


def tokenize_dataset(DATA_PATH):
    load_dotenv()

    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")

    # df = pd.read_csv(DATA_PATH)
    df = pd.read_csv(
        DATA_PATH, storage_options={"key": aws_access_key, "secret": aws_secret_key}
    )

    df = preprocess_data(df)

    all_reviews_list = df["review"].tolist()

    encoded_labels = np.array([0 if x == "negative" else 1 for x in df["sentiment"]])

    return all_reviews_list, encoded_labels


if __name__ == "__main__":
    DATA_PATH = os.path.abspath(sys.argv[1])
