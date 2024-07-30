import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import joblib

# Load the dataset
# df = pd.read_csv("C:/Users/harsh/OneDrive/Desktop/projects/Flaskapp-SentimentAnalysis/Datatset/Twitter_Data.csv", header=None, names=["text", "target"])
df = pd.read_csv("C:/Users/harsh/OneDrive/Desktop/projects/Flaskapp-SentimentAnalysis/Datatset/Twitter_Data.csv", header=None, names=["text", "target"])

df["text"] = df["text"].replace(np.nan, "", regex=True)

def normalize_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])
        text = text.strip()
    return text

df["text"] = df["text"].apply(normalize_text)

# Drop rows with NaN values in the target column
df = df.dropna(subset=["target"])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["target"], test_size=0.2, random_state=42)

# Create TF-IDF vectors
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Train an SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train_tfidf, y_train)

# Save the model and vectorizer
joblib.dump(svm_classifier, 'svm_classifier.pkl')
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')
print(f'Model saved.')
