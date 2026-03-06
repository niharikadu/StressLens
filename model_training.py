from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)

# Load trained model files
model = joblib.load("model.pkl")
tfidf = joblib.load("tfidf.pkl")
label_encoder = joblib.load("label_encoder.pkl")


@app.route("/")
def home():
    return "AI Exam Anxiety Detector API Running"


@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()
    text = data["text"]

    # Convert text to vector
    vector = tfidf.transform([text])

    # Predict
    prediction = model.predict(vector)[0]
    label = label_encoder.inverse_transform([prediction])[0]

    return jsonify({
        "anxiety_level": label
    })


if __name__ == "__main__":
    app.run(debug=True)