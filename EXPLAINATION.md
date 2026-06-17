# Word-to-Word Project Explanation: AI Resume Analyzer

This document provides a detailed, step-by-step, file-by-file explanation of how the entire AI Resume Analyzer project works.

## 1. `app.py` - The Flask Backend (The Application Core)
The `app.py` file is the heart of the web application. It connects the user interface (HTML) to the Artificial Intelligence models in the background.

- **Setup & Configuration**: It starts by importing Flask, the AI `MLPipeline` from `ml_pipeline.py`, and helper functions (`parse_pdf`, `clean`) from `utils.py`. It also creates an `uploads` folder to temporarily store PDF files that users upload.
- **Pipeline Initialization**: `pipeline = MLPipeline(ds_path='resume_data.csv')` initializes the machine learning models and trains them instantly using the historical data inside `resume_data.csv`.
- **Home Route (`@app.route('/')`)**: When you visit the main URL, it loads `index.html`, which is the user interface where you can upload a resume or paste text.
- **Analyze Route (`@app.route('/analyze')`)**: When a user submits a resume:
  1. It checks if a file was uploaded. If it's a PDF, it uses `parse_pdf()` to extract the raw text. If it's a text file, it decodes it directly. Alternatively, if the user pasted raw text, it grabs that.
  2. If no text is found, it returns an error message.
  3. The raw text is passed to the `clean()` function, which removes unnecessary characters, special symbols, and stop-words.
  4. The cleaned text is sent to the AI: `pipeline.predict(cleaned)`. The AI returns a dictionary containing the predicted job category, an interview score, a cluster ID, and extracted keywords.
  5. **Self-Learning Step**: `pipeline.learn(cleaned, result['category'], result['score'])` is called. This appends the newly analyzed resume back into the `resume_data.csv` file. This means the dataset grows, and the AI gets smarter over time.
  6. Finally, the results are sent to `result.html` to be displayed nicely to the user.

## 2. `ml_pipeline.py` - The AI Brain (Machine Learning Pipeline)
This file contains the `MLPipeline` class, which handles all the AI, machine learning, and natural language processing (NLP).

- **Initialization (`__init__`)**: It creates instances of the machine learning algorithms:
  - `TfidfVectorizer`: Converts words into numbers (features). It focuses on the most important words (max 1000 features) and ignores common English "stop words" (like 'and', 'the').
  - `MultinomialNB`: A Naive Bayes classifier used to predict the job category.
  - `RandomForestRegressor`: A regression model that predicts a score from 0 to 100.
  - `KMeans`: A clustering algorithm that groups resumes into 5 distinct groups based on similarities.
- **The `setup()` Method**: This runs automatically when the app starts. 
  1. It reads `resume_data.csv` using Pandas.
  2. It cleans the data (dropping empty rows).
  3. **Crucial Step**: It balances the dataset `data.groupby('Job_Category').sample(n=30, replace=True)`. This ensures that the AI doesn't become biased toward job categories that have more data.
  4. It uses `fit_transform` to train the TF-IDF vectorizer on the text.
  5. It trains (`fit`) the Naive Bayes model to predict job categories, the K-Means model to cluster, and the Random Forest model to predict interview scores.
- **The `predict(txt)` Method**: When a new resume is analyzed:
  1. It converts the new text into numbers using the already-trained TF-IDF vectorizer (`transform`).
  2. It asks the Naive Bayes model to predict the category.
  3. It asks the K-Means model to assign a cluster.
  4. It asks the Random Forest model to predict the score.
  5. It calls `extract_keywords()` to find the 10 most prominent skills/words in the text based on TF-IDF scoring.
  6. Returns all this data as a neat dictionary.
- **The `learn()` Method**: Opens `resume_data.csv` in 'append' mode (`'a'`). It creates a new unique ID based on the current timestamp and adds a new row with the analyzed text, predicted category, and score. This is how the AI "learns" from new inputs.

## 3. `utils.py` - Helper Functions
Though small, this file handles text extraction and cleaning.
- `parse_pdf()`: Uses `PyPDF2` to open a PDF file, go through every page, and extract all the text into a single large string.
- `clean()`: Uses Regular Expressions (`re`) or NLP libraries to strip out email addresses, phone numbers, punctuation, extra spaces, and special characters so the AI models only focus on the core skills and experience.

## 4. `resume_data.csv` - The Knowledge Base
This is a massive dataset (over 8MB) containing thousands of real-world resumes. Each row has the original resume text, the actual job category, and a baseline interview score. The AI uses this as its "textbook" to learn how to identify software engineers, HR professionals, data scientists, etc.

## Summary Flow
1. User Uploads Resume (PDF) -> `app.py`
2. PDF is converted to text -> `utils.py`
3. Text is cleaned -> `utils.py`
4. Clean text is given to AI -> `ml_pipeline.py`
5. AI converts words to numbers, predicts category, score, cluster, and keywords.
6. AI saves the new resume to `resume_data.csv` to learn.
7. Results are shown to the user on the screen.
