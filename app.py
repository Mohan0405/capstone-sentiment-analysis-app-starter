import os
import pickle
from pathlib import Path
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Global variables
model = None
tokenizer = None
analyzer = SentimentIntensityAnalyzer()

# Define absolute paths
base_dir = Path(__file__).resolve().parent
model_path = base_dir / "models" / "uci_sentimentanalysis.h5"
tokenizer_path = base_dir / "models" / "tokenizer.pickle"

# Load Keras model
def load_keras_model():
    global model
    if model is None:
        print(f"🔎 Checking model path: {model_path}")
        if not model_path.exists():
            print("❌ Model file not found!")
            return
        model = load_model(model_path)
        print("✅ Keras model loaded.")

# Load tokenizer
def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        print(f"🔎 Checking tokenizer path: {tokenizer_path}")
        if not tokenizer_path.exists():
            print("❌ Tokenizer file not found!")
            return
        with open(tokenizer_path, 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✅ Tokenizer loaded.")

# Sentiment analysis using custom model
def sentiment_analysis(text):
    global tokenizer, model
    load_tokenizer()
    load_keras_model()
    
    if tokenizer is None or model is None:
        raise Exception("Model or tokenizer not loaded properly.")

    user_sequences = tokenizer.texts_to_sequences([text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

# Main route
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text = ""
    error = None
    try:
        if request.method == "POST":
            text = request.form.get("text", "")
            print("📨 Input received:", text)

            vader_result = analyzer.polarity_scores(text)
            print("🧠 VADER result:", vader_result)

            vader_result["custom model positive"] = sentiment_analysis(text)
            sentiment = vader_result
    except Exception as e:
        error = str(e)
        print("❌ Full Exception:", e)

    return render_template("form.html", sentiment=sentiment, text=text, error=error)

# Health check endpoint
@app.route("/health")
def health():
    return jsonify({
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

# Required for local and Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
