import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
import os
import csv

class MLPipeline:
    def __init__(self, ds_path='resume_data.csv'):
        self.ds_path = ds_path
        self.vec = TfidfVectorizer(stop_words='english', max_features=1000)
        self.clf = MultinomialNB()
        self.reg = RandomForestRegressor(n_estimators=50, random_state=42)
        self.km = KMeans(n_clusters=5, random_state=42)
        
        self.feature_names = []
        self.is_trained = False
        self.setup()

    def setup(self):
        if not os.path.exists(self.ds_path):
            return

        data = pd.read_csv(self.ds_path)
        if data.empty:
            return
            
        # Clean data to prevent NaN crashes
        data = data.dropna(subset=['Resume_Text', 'Job_Category'])
        if 'Interview_Score' in data.columns:
            data['Interview_Score'] = data['Interview_Score'].fillna(80)
            
        # Balance dataset to fix Naive Bayes class imbalance bias!
        data = data.groupby('Job_Category').sample(n=30, replace=True)
            
        X_txt = self.vec.fit_transform(data['Resume_Text'])
        self.feature_names = self.vec.get_feature_names_out()
        
        y_cat = data['Job_Category']
        self.clf.fit(X_txt, y_cat)

        self.km.fit(X_txt)

        if 'Interview_Score' in data.columns:
            y_sc = data['Interview_Score']
        else:
            y_sc = np.random.randint(60, 100, size=len(data))
            
        self.reg.fit(X_txt, y_sc)
        self.is_trained = True

    def extract_keywords(self, tfidf_matrix):
        """Extract top 10 keywords based on TF-IDF scores."""
        scores = tfidf_matrix.toarray()[0]
        top_indices = scores.argsort()[-10:][::-1]
        keywords = [self.feature_names[i] for i in top_indices if scores[i] > 0]
        return keywords

    def predict(self, txt):
        if not self.is_trained:
            raise Exception("Models are not trained yet.")
            
        txt_feats = self.vec.transform([txt])
        
        pred_cat = self.clf.predict(txt_feats)[0]
        pred_clus = self.km.predict(txt_feats)[0]
        pred_sc = self.reg.predict(txt_feats)[0]
        
        keywords = self.extract_keywords(txt_feats)
        
        return {
            "category": pred_cat,
            "cluster": int(pred_clus),
            "score": round(pred_sc, 2),
            "keywords": keywords
        }

    def learn(self, txt, category, score):
        """Appends the new resume to the global dataset to learn over time."""
        with open(self.ds_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # Columns: ID, Job_Category, Resume_Text, Interview_Score
            import time
            new_id = int(time.time())
            writer.writerow([new_id, category, txt, score])
            # Trigger reload 2
