#!/bin/bash

echo "=========================================="
echo "    AI Resume Analyzer Setup & Startup    "
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "-> Activating virtual environment..."
    source venv/bin/activate
else
    echo "-> No 'venv' found, proceeding with system/user python environment..."
fi

echo "-> Installing required packages from requirements.txt..."
# Including --break-system-packages to bypass OS restrictions
pip install -r requirements.txt --break-system-packages

echo "-> Downloading required NLTK Data..."
python3 -c "import nltk; nltk.download('stopwords', quiet=True); nltk.download('punkt', quiet=True); nltk.download('wordnet', quiet=True)"

echo "-> Starting the Flask Application..."
python3 app.py
