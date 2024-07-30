from flask import Flask, render_template, request
import joblib
import re
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Load the saved model and vectorizer
svm_classifier = joblib.load('svm_classifier.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Preprocess the user input text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
    text = text.strip()
    return text

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    if request.method == "POST":
        user_input = request.form["user_input"]
        preprocessed_input = preprocess_text(user_input)
        vector = tfidf_vectorizer.transform([preprocessed_input])
        sentiment = svm_classifier.predict(vector)[0]  
    return render_template("index.html", sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
