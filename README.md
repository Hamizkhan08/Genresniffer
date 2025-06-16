
# 🎧 GenreSniffer

GenreSniffer is a web-based machine learning application that predicts the genre of a song or audio snippet using textual input features. It uses NLP techniques and a trained classification model to identify genres from song-related data.

## 🗂 Project Structure

```

Genresniffer-main/
│
├── app.py                    # Flask app for the web interface
├── train\_model.py            # Script to train and save the model
├── test.csv                  # Test dataset
├── cleaned\_test.csv          # Cleaned version of the test dataset
├── feedback\_data.csv         # Optional dataset for user feedback
│
├── model/
│   ├── model.pkl             # Trained ML model
│   ├── tfidf.pkl             # TF-IDF vectorizer
│   └── labelencoder.pkl      # Label encoder for genre labels
│
├── templates/
│   └── index.html            # HTML template for the web UI
│
├── static/
│   └── style.css             # CSS for styling the web page
│
├── requirements.txt          # List of Python dependencies
└── runtime.txt               # Runtime specification for deployment

````

## 🖼️ Screenshot

Here’s how the GenreSniffer web interface looks:

![App Interface](images/screenshot_home.png)

> 💡 Place your screenshot at: `Genresniffer-main/images/screenshot_home.png`

## 🚀 Features

- Genre prediction from textual input (e.g., lyrics, metadata)
- Simple web interface built using Flask
- Trained model using NLP + ML pipeline
- TF-IDF vectorization and Label Encoding
- Ready for local or cloud deployment

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Genresniffer.git
   cd Genresniffer-main
   ```

2. **Create a virtual environment** (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install the dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

## 💻 Usage

1. **Run the Flask app**:

   ```bash
   python app.py
   ```

2. **Open in browser**:
   [http://127.0.0.1:5000](http://127.0.0.1:5000)

3. **Input song-related text** (e.g., title, artist, lyrics, description) and get the predicted genre instantly.

## 🧠 Model Training

To retrain the model on your own data:

```bash
python train_model.py
```

Make sure your input CSV is formatted appropriately and update paths if needed.

## 🌐 Deployment

* Designed for easy deployment on platforms like Heroku, Render, or Replit.
* `runtime.txt` and `requirements.txt` are included to support deployment.

## 📊 Data

* `test.csv`, `cleaned_test.csv`: Datasets used for evaluation or testing.
* `feedback_data.csv`: Optional user feedback or extra training data.

## ✨ Future Enhancements

* Add audio file support using audio feature extraction (MFCC, etc.)
* Collect live feedback and retrain model periodically
* Improve accuracy with larger and richer datasets

## 📃 License

This project is for educational use. Feel free to fork and modify.

## 👤 Author

* Hamiz Khan

