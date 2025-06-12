# main.py
from flask import Flask, request, jsonify
from analyse import analyze_dataset
import os
import tempfile

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    target_word = request.form.get("target_word")
    if not target_word or 'audio_files' not in request.files:
        return jsonify({"error": "Missing target_word or audio_files"}), 400

    # Save uploaded files
    with tempfile.TemporaryDirectory() as temp_dir:
        for audio in request.files.getlist("audio_files"):
            audio.save(os.path.join(temp_dir, audio.filename))

        result = analyze_dataset(target_word, temp_dir)
        return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
