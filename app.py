"""
ExamEase — Flask App (Fully Integrated Version)
Flask serves BOTH the frontend HTML AND the API.
No CORS issues. No file:// problems. Everything on one port.
"""
import os
import scipy.sparse as sp
import numpy as np
import joblib
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from preprocess import clean_text, PHYSIO_FEATURES

# ── App setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)
CORS(app)

# ── Load model artifacts ─────────────────────────────────────────────────────
MODEL_DIR = os.path.join(BASE_DIR, "models")

try:
    model  = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
    tfidf  = joblib.load(os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    print("✅ Model artifacts loaded successfully.")
except Exception as e:
    print(f"❌ ERROR loading models: {e}")
    print("   Run 'python train_model.py' first!")
    model = tfidf = scaler = None

LABELS = ["Low", "Moderate", "High"]

RECOMMENDATIONS = {
    "Low": {
        "message": "You are managing exam pressure well! 🌟",
        "tips": [
            "Maintain your current study schedule",
            "Continue regular sleep and exercise habits",
            "Practice mindfulness to stay grounded",
            "Reward yourself for consistent preparation",
        ],
        "color": "#2ecc71", "emoji": "😊"
    },
    "Moderate": {
        "message": "You are experiencing manageable exam stress. ⚡",
        "tips": [
            "Try the Pomodoro technique (25 min study, 5 min break)",
            "Practice deep breathing exercises daily",
            "Reduce caffeine and improve sleep hygiene",
            "Talk to a friend or family member about your concerns",
            "Break your syllabus into smaller daily targets",
        ],
        "color": "#f39c12", "emoji": "😐"
    },
    "High": {
        "message": "You may be under significant exam anxiety. Please seek support. 🆘",
        "tips": [
            "Speak to your college counselor immediately",
            "Practice progressive muscle relaxation",
            "Take a complete break from studies for one day",
            "Call iCall helpline: 9152987821",
            "Remember: one exam does not define your future",
        ],
        "color": "#e74c3c", "emoji": "😰"
    }
}

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Serve the frontend HTML page."""
    return render_template("index.html")

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "app": "ExamEase v1.0"
    })

@app.route("/api/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded. Run train_model.py first."}), 500

    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        # Validate fields
        required = ["text_response"] + PHYSIO_FEATURES
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({"error": f"Missing fields: {missing}"}), 400

        # Build DataFrame
        row = {
            "text_response": str(data["text_response"]),
            **{feat: float(data[feat]) for feat in PHYSIO_FEATURES}
        }
        df_row = pd.DataFrame([row])

        # Preprocess text
        cleaned  = df_row["text_response"].apply(clean_text)
        text_vec = tfidf.transform(cleaned)

        # Preprocess numerics
        num_data   = df_row[PHYSIO_FEATURES].fillna(0)
        num_scaled = scaler.transform(num_data)
        num_sparse = sp.csr_matrix(num_scaled)

        # Combine & predict
        X          = sp.hstack([text_vec, num_sparse])
        pred_code  = int(model.predict(X)[0])
        proba      = model.predict_proba(X)[0].tolist()
        label      = LABELS[pred_code]
        confidence = round(proba[pred_code] * 100, 2)
        rec        = RECOMMENDATIONS[label]

        return jsonify({
            "anxiety_level": label,
            "anxiety_code":  pred_code,
            "confidence":    confidence,
            "probabilities": {
                "Low":      round(proba[0] * 100, 2),
                "Moderate": round(proba[1] * 100, 2),
                "High":     round(proba[2] * 100, 2),
            },
            "recommendation": rec["message"],
            "tips":           rec["tips"],
            "color":          rec["color"],
            "emoji":          rec["emoji"],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 StressLens AI running at → http://127.0.0.1:{port}")
    print(f"   Open this URL in your browser!\n")
    # app.run(host="0.0.0.0", port=port, debug=True)