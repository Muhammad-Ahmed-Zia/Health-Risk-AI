"""
app.py  –  Health Risk Prediction System  –  Flask backend
Run:  python app.py
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR    = os.path.join(BASE_DIR, "static")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
CORS(app)


# ── Page routes ────────────────────────────────────────────────────────────────
@app.route("/")
def welcome():
    return send_from_directory(TEMPLATES_DIR, "welcome.html")


@app.route("/form")
def form():
    return send_from_directory(TEMPLATES_DIR, "form.html")


@app.route("/result")
def result():
    return send_from_directory(TEMPLATES_DIR, "result.html")


# ── Static assets (css, js, images) ───────────────────────────────────────────
@app.route("/static/<path:path>")
def static_files(path):
    return send_from_directory(STATIC_DIR, path)


# ── API ────────────────────────────────────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        from predict import predict_risk
        data = request.get_json(force=True)

        required = [
            "first_name", "last_name",
            "age", "weight", "height",
            "exercise", "sleep", "sugar_intake",
            "smoking", "alcohol", "married", "profession",
        ]
        missing = [k for k in required if k not in data or str(data[k]).strip() == ""]
        if missing:
            return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

        result = predict_risk(data)
        result["first_name"] = data["first_name"].strip()
        result["last_name"]  = data["last_name"].strip()
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
