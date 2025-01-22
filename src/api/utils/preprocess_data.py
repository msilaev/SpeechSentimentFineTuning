import re
import nltk
from nltk.corpus import stopwords
from bs4 import BeautifulSoup

from string import punctuation

# from dotenv import load_dotenv

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


def preprocess_text(text):
    stop = set(stopwords.words("english"))
    stop.discard("not")

    text = text.lower()
    text = remove_between_square_brackets(text)
    text = denoise_text(text)
    text = "".join([c for c in text if c not in punctuation])
    text = " ".join([c for c in text.split() if c not in stop])

    return text
