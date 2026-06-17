from flask import Flask, render_template, request
import os
from utils import parse_pdf, clean
from ml_pipeline import MLPipeline

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

pipeline = MLPipeline(ds_path='resume_data.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def process():
    res_text = ""
    if 'file' in request.files and request.files['file'].filename != '':
        f = request.files['file']
        if f.filename.endswith('.pdf'):
            res_text = parse_pdf(f)
        else:
            res_text = f.read().decode('utf-8', errors='ignore')
    elif request.form.get('raw_text'):
        res_text = request.form.get('raw_text')

    if not res_text.strip():
        return render_template('index.html', err="Please upload a PDF or enter text.")

    cleaned = clean(res_text)

    try:
        result = pipeline.predict(cleaned)
        # Self-learning: append the new resume to the dataset
        pipeline.learn(cleaned, result['category'], result['score'])
    except Exception as e:
        return render_template('index.html', err=f"Processing failed: {e}")

    return render_template('result.html', 
                           category=result['category'],
                           score=result['score'],
                           cluster=result['cluster'],
                           keywords=result['keywords'])

if __name__ == '__main__':
    app.run(debug=True)
