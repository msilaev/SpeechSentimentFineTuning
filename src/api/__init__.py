from .config import Config
from flask import Flask
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

app = Flask(__name__)
app.config.from_object(Config)
from src.api import routes
