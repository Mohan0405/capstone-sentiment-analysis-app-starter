import os
import pickle
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize Flask
app = Flask(__name__)

# Global variables
model = None
tokenizer = None
analyzer = SentimentIntensityAnalyzer()

# Load Keras model
def load_keras_model():
    global model
    if model is None:
        model = load_model('models/uci_sentimentanalysis.h5')
        print("✅ Keras model loaded.")

# Load tokenizer
def load_tokenizer():
    global tokenizer
    if tokenizer is None:
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✅ Tokenizer loaded.")

# Sentiment analysis using custom model
def sentiment_analysis(text):
    global tokenizer, model
    # Ensure model and tokenizer are loaded
    load_tokenizer()
    load_keras_model()

    user_sequences = tokenizer.texts_to_sequences([text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

# Main route: Form to enter text
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

# Health check endpoint for debugging
@app.route("/health")
def health():
    return jsonify({
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

# Required for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
