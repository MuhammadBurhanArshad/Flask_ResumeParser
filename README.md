# AI Resume Analyzer

An intelligent, self-learning web application that automates resume screening, keyword extraction, and candidate analysis using Machine Learning. Built with Flask, Pandas, Scikit-Learn, and various AI models (Random Forest, Naive Bayes, Decision Tree, K-Means Clustering, and KNN).

## 🚀 Project Overview

The AI Resume Analyzer solves the real-world problem of manual resume screening by automating the extraction of data from PDF resumes or raw text. By applying robust Machine Learning algorithms, the application can extract top skills (using TF-IDF), classify the candidate's professional domain, predict an "Interview Probability Score", and group similar profiles together.

### 📸 Application Screenshots

<img width="1349" height="691" alt="image" src="https://github.com/user-attachments/assets/7f61b1e8-f16a-4a67-a5a6-ef7b6df9d1e2" />

<img width="1349" height="867" alt="image" src="https://github.com/user-attachments/assets/8214cf9c-4bb3-4f1d-9ffe-f0c3460e0c38" />


### 🧠 AI Integrations & Algorithms Used
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Used to dynamically extract the top 10 keywords and skills from the resume text without relying on manual, static keyword lists.
- **Multinomial Naive Bayes**: Used for text classification, predicting the domain or job category of a resume based on the extracted and vectorized text. The dataset is balanced during setup to prevent class imbalance bias.
- **K-Means Clustering**: Unsupervised learning used to group similar candidates together into 5 distinct clusters (e.g., grouping all frontend developers into a single cluster).
- **Random Forest Regressor**: An ensemble model used to predict a candidate's "Interview Probability Score" (from 0 to 100) based on the textual features.
- **AI Self-Learning Pipeline**: The system continuously improves itself. Every time a new resume is analyzed, the parsed text, predicted category, and score are appended to the global dataset (`resume_data.csv`).

### 🛠️ Technology Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn (TfidfVectorizer, MultinomialNB, RandomForestRegressor, KMeans)
- **Data Handling**: Pandas, NumPy, CSV
- **Document Processing**: PyPDF2 (via custom utils)
- **Frontend**: HTML, CSS (Bootstrap via Templates)

---

## 💻 Project Setup Instructions

Follow these steps to run the project locally on your machine.

### 1. Navigate to the Project Directory
Open your terminal and navigate to this folder:
```bash
cd /Flask_ResumeParser
```

### 2. Activate the Virtual Environment
A virtual environment keeps your project dependencies isolated.
```bash
source venv/bin/activate
```
*(You should see `(venv)` appear at the start of your terminal line).*

### 3. Install Dependencies
Install all required libraries such as Flask, Pandas, and Scikit-Learn:
```bash
pip install -r requirements.txt
```

### 4. Run the Application
Start the Flask development server:
```bash
python3 app.py
```
Open your web browser and go to `http://127.0.0.1:5000` to interact with the application.

---

## 📂 Expected Directory Structure
```
Flask_ResumeParser/
│
├── app.py                # Main Flask application and API routes
├── ml_pipeline.py        # Core AI/ML pipeline, training, and prediction logic
├── utils.py              # Helper functions (PDF parsing, text cleaning)
├── resume_data.csv       # Global dataset for training and self-learning
├── requirements.txt      # Python dependencies
├── templates/            # HTML interface (index.html, result.html)
├── static/               # CSS and JS files
└── uploads/              # Temporary folder for uploaded PDFs
```
