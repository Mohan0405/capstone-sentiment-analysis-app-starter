import os
import pickle
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

# Load Keras sentiment model
def load_keras_model():
    global model
    try:
        model = load_model('models/uci_sentimentanalysis.h5')
        print("✅ Model loaded.")
    except Exception as e:
        print("❌ Failed to load model:", e)

# Load tokenizer
def load_tokenizer():
    global tokenizer
    try:
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✅ Tokenizer loaded.")
    except Exception as e:
        print("❌ Failed to load tokenizer:", e)

# Load everything once at first request
@app.before_first_request
def before_first_request():
    load_keras_model()
    load_tokenizer()

# Custom sentiment prediction
def sentiment_analysis(text):
    user_sequences = tokenizer.texts_to_sequences([text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

# Main route (GET for form, POST for result)
@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        vader_result = analyzer.polarity_scores(text)
        vader_result["custom model positive"] = sentiment_analysis(text)
        sentiment = vader_result
    return render_template("form.html", sentiment=sentiment, text=text)

# Health check endpoint (useful for Render or uptime monitoring)
@app.route("/health")
def health():
    return jsonify({
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

# Local development server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
