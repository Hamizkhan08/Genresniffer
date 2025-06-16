import pandas as pd
import re
import string
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("test.csv")
df = df[['Artist', 'Song', 'Genre', 'Lyrics']].dropna()
df = df.rename(columns={"Lyrics": "lyrics", "Genre": "genre"})

# -----------------------------
# Text Preprocessing
# -----------------------------
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_lyrics(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words]
    return " ".join(tokens)

print("🔄 Cleaning lyrics... (may take a few minutes)")
df['clean_lyrics'] = df['lyrics'].apply(clean_lyrics)

# -----------------------------
# TF-IDF Vectorization
# -----------------------------
print("⚙️ Vectorizing lyrics...")
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['clean_lyrics'])

# -----------------------------
# Label Encoding
# -----------------------------
le = LabelEncoder()
y = le.fit_transform(df['genre'])

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
print("🚀 Training model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
print("📊 Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy: {accuracy:.4f}")
print("\n📌 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -----------------------------
# Save model and tools
# -----------------------------
os.makedirs("model", exist_ok=True)
pickle.dump(model, open("model/model.pkl", "wb"))
pickle.dump(tfidf, open("model/tfidf.pkl", "wb"))
pickle.dump(le, open("model/labelencoder.pkl", "wb"))
df[['lyrics', 'genre', 'clean_lyrics']].to_csv("cleaned_test.csv", index=False)

print("✅ Model, TF-IDF, and LabelEncoder saved to 'model/'")
print("✅ Cleaned lyrics saved to 'cleaned_test.csv'")
