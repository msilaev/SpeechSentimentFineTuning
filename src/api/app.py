import sys
import os

# Add the src directory to the Python path
# sys.path.append(os.path.join(os.path.dirname(__file__), "src", "api"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.api import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
    print("app initialized")
