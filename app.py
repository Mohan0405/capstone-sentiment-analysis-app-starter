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
    model = load_model('models/uci_sentimentanalysis.h5')
    print("✅ Keras model loaded.")

# Load tokenizer
def load_tokenizer():
    global tokenizer
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    print("✅ Tokenizer loaded.")

# Sentiment analysis using custom model
def sentiment_analysis(text):
    user_sequences = tokenizer.texts_to_sequences([text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    return round(float(prediction[0][0]), 2)

# Main route: Form to enter text
@app.route("/", methods=["GET", "POST"])
def index():
    global model, tokenizer

    # Lazy-load model and tokenizer
    if model is None:
        load_keras_model()
    if tokenizer is None:
        load_tokenizer()

    sentiment = None
    text = ""
    if request.method == "POST":
        text = request.form.get("text", "")
        vader_result = analyzer.polarity_scores(text)
        vader_result["custom model positive"] = sentiment_analysis(text)
        sentiment = vader_result

    return render_template("form.html", sentiment=sentiment, text=text)

# Health check endpoint for debugging
@app.route("/health")
def health():
    return jsonify({
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None
    })

#if __name__ == "__main__":
#    # Local development only. Use gunicorn for Render.
#    app.run(host="0.0.0.0", port=5000, debug=True)
