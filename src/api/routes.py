import sys
import os
import subprocess
import librosa as lb
import openai
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.api.utils.predict import PredictSentiment
from src.api import app

from flask import (
    render_template,
    request,
    url_for,
    jsonify,
    redirect 
)

param_yaml_path = os.path.join(
    os.path.
    dirname(os.path.dirname(app.root_path)), app.config["PARAM_YAML_PATH"]
)

load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")
classifier = PredictSentiment(param_yaml_path)

@app.route("/", methods=["GET", "POST"])
def start():
    #print(request.method)
    if request.method == "POST":
        #print(request.form)
        
        if 'text' in request.form:
            text = request.form['text']
           # print(text)
            return redirect(url_for("text_upload", text = text))
        
        elif 'file' in  request.files:
            file = request.files['file']            


            if not file or file.filename == '' or (not file.filename.endswith('.mp3')):
                return jsonify({'error': 'Please upload a mp3 file'})
            
            max_file_size_MB = 30

            max_file_size = max_file_size_MB * 1024 * 1024  # 10 MB (adjust this as needed)

            try:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename))

                audio = os.path.join(app.root_path, app.config['UPLOAD_FOLDER'], filename)
                file_size = os.path.getsize(audio)

                if file_size > max_file_size:
                    return jsonify({ f'error': "File size exceeds the limit of {max_file_size_MB} MB!"})

                # Open an audio file
                with open(audio, "rb") as audio_file:
                    transcript = openai.Audio.transcribe("whisper-1", audio_file)
                text_input = str(transcript.get('text', ''))
                try:
                    upload_folder = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"])
                    os.makedirs(upload_folder, exist_ok=True)
                    file_path = os.path.join(upload_folder, "user.txt")

                    with open(file_path, "w", encoding="utf-8") as file:
                        file.write(text_input)

                    return redirect(url_for("predict"))

                except Exception as e:
                    return jsonify({"error": str(e)})
                
            except Exception as e:
                return jsonify({'error': str(e)})

    return render_template("upload.html")


#########################################
@app.route("/upload_text", methods=["GET", "POST"])
def text_upload():
    #print("upload", app.root_path)

        text_input = request.args.get("text", "").strip()

        if not text_input:
            return jsonify({"error": "Please enter some text"})

        try:
            upload_folder = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"])
            os.makedirs(upload_folder, exist_ok=True)
            file_path = os.path.join(upload_folder, "user.txt")

            with open(file_path, "w", encoding="utf-8") as file:
                file.write(text_input)

            return redirect(url_for("predict"))

        except Exception as e:
            return jsonify({"error": str(e)})

        return render_template("upload.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    print(app.root_path)
    try:
        # Define the file path
        file_name = os.path.join(app.root_path, app.config["UPLOAD_FOLDER"], "user.txt")

        # Ensure the file exists
        if not os.path.exists(file_name):
            return (
                jsonify({"error": "No uploaded text found. Please upload text first."}),
                400,
            )

        # Read the text from the file
        with open(file_name, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Validate the content
        if not text:
            return (
                jsonify(
                    {"error": "The uploaded file is empty. Please upload valid text."}
                ),
                400,
            )

        # Make prediction using the classifier
        prediction = classifier.predict(text)
        sentiment = "Positive" if prediction[0] == 1 else "Negative"

        # Render the result in the report.html template
        return render_template("report.html", result=(sentiment, text))

    except Exception as e:
        # Handle any unexpected errors
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

@app.route("/report/<result>")
def report(result):
    return render_template("report.html", result=result)
