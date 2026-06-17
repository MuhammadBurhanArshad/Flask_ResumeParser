# AI Resume Analyzer

An intelligent web application that automates resume screening and candidate analysis using Machine Learning. Built for an academic project utilizing Flask, Pandas, and various AI models (Random Forest, Naive Bayes, Decision Tree, K-Means Clustering, and KNN).

## 🚀 Project Overview

This Application-Based Project solves the real-world problem of manual resume screening. By applying Machine Learning algorithms, the application can extract features from resumes, classify candidates, and group similar profiles together.

### 🧠 AI Integrations & Algorithms Used
- **Pandas & NumPy**: For dataset manipulation, cleaning, and preprocessing candidate data.
- **Naive Bayes / Decision Tree**: Used for text classification (e.g., predicting the domain or job category of a resume based on extracted text).
- **K-Means Clustering**: Unsupervised learning used to group similar candidates together (e.g., grouping all frontend developers into a single cluster).
- **KNN (K-Nearest Neighbors)**: Used to find the top 'K' candidate resumes that are most similar to a given job description.
- **Random Forest**: An ensemble model used to predict a candidate's "Interview Probability Score" based on various features (years of experience, number of skills matched, education level).

### 🛠️ Technology Stack
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-Learn, NLTK
- **Data Handling**: Pandas
- **Document Processing**: PyPDF2
- **Frontend**: HTML, CSS (Bootstrap)

---

## 💻 Project Setup Instructions

Follow these steps to run the project locally on your machine.

### 1. Navigate to the Project Directory
Open your terminal and navigate to this folder:
```bash
cd ~/Practice/AI/resume_parser/ResumeAnalyzer
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

### 4. Download NLP Data (If using NLTK)
If the AI models require natural language processing to clean the resume text:
```bash
python3 -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('wordnet')"
```

### 5. Run the Application
Start the Flask development server:
```bash
python3 app.py
```
Open your web browser and go to `http://127.0.0.1:5000` to interact with the application.

---

## 📂 Expected Directory Structure
As you build the project, your folder should look like this:
```
ResumeAnalyzer/
│
├── app.py                # Main Flask application
├── models/               # Saved ML models (.pkl files)
├── templates/            # HTML files for the web interface
├── static/               # CSS and JS files
├── uploads/              # Temporary folder for uploaded PDFs
├── requirements.txt      # List of dependencies
├── .gitignore            # Git ignore rules
└── venv/                 # Virtual Environment
```
