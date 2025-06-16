# 1. IMPORTS
import os
import csv
from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from urllib.parse import quote_plus

# --- One-time setup ---
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

download_nltk_data()


# 2. FLASK APP INITIALIZATION
app = Flask(__name__)
PROJECT_NAME = "GenreSniffer"  # Define project name here

# 3. LOAD MODELS & DATA
model = pickle.load(open("model/model.pkl", "rb"))
tfidf = pickle.load(open("model/tfidf.pkl", "rb"))
le = pickle.load(open("model/labelencoder.pkl", "rb"))
FEEDBACK_FILE = 'feedback_data.csv'

# 4. HELPER FUNCTION
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()
def clean_lyrics(text):
    text = re.sub(r'\[.*?\]', '', text); text = re.sub(r'\n', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text); text = re.sub(r'<.*?>+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation)); text = re.sub(r'\w*\d\w*', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [stemmer.stem(t) for t in tokens if t not in stop_words and t.isalpha()]
    return " ".join(tokens)

# 5. FLASK ROUTES
@app.route('/')
def home():
    return render_template("index.html", project_name=PROJECT_NAME)

@app.route('/predict', methods=["POST"])
def predict():
    input_lyrics = request.form["lyrics"]
    
    first_two_lines = " ".join(input_lyrics.splitlines()[:2])
    identify_song_google_link = f"https://www.google.com/search?q={quote_plus(f'{first_two_lines} lyrics')}"

    cleaned_input = clean_lyrics(input_lyrics)
    vectorized_input = tfidf.transform([cleaned_input])
    probabilities = model.predict_proba(vectorized_input)[0]
    
    top_indices = np.argsort(probabilities)[::-1][:5]
    top_genres = [{'genre': le.classes_[i], 'probability': probabilities[i]} for i in top_indices]

    all_genres = sorted(le.classes_)

    return render_template(
        "index.html",
        project_name=PROJECT_NAME,
        lyrics_input=input_lyrics,
        top_genres=top_genres,
        all_genres=all_genres,
        identify_song_google_link=identify_song_google_link
    )

@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.form
        lyrics = data.get('lyrics_for_feedback')
        predicted_genre = data.get('predicted_genre')
        corrected_genre = data.get('corrected_genre')
        is_correct = data.get('is_correct')

        if is_correct == 'true':
            feedback_status = 'correct'
            final_genre = predicted_genre
        else:
            feedback_status = 'incorrect'
            final_genre = corrected_genre

        file_exists = os.path.isfile(FEEDBACK_FILE)
        with open(FEEDBACK_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['lyrics', 'predicted_genre', 'final_genre', 'feedback_status'])
            writer.writerow([lyrics, predicted_genre, final_genre, feedback_status])
        
        return jsonify({'status': 'success', 'message': 'Feedback received. Thank you!'})

    except Exception as e:
        print(f"Error saving feedback: {e}")
        return jsonify({'status': 'error', 'message': 'Could not save feedback.'}), 500

if __name__ == '__main__':
    app.run(debug=True)
