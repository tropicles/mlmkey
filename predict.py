from sklearn.feature_extraction.text import TfidfVectorizer
from preprocess import preprocess
import numpy as np
import pickle
import os
import joblib

class KeywordExtractor:
    def __init__(self):
        self.model_path = "tfidf_model.pkl"
        # Load model if it exists, otherwise create a new one
        if os.path.exists(self.model_path):
            self.vectorizer = joblib.load(self.model_path)
        else:
            self.vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=10000,
                stop_words='english'
            )
            # The vectorizer will be fit on first use
    
    def extract_keywords(self, job_description, num_keywords=15):
        processed_text = preprocess(job_description)
        
        # If the vectorizer hasn't been fit yet, fit it on this text
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit([processed_text])
            joblib.dump(self.vectorizer, self.model_path)
        
        # Transform text to a TF-IDF vector
        tfidf_matrix = self.vectorizer.transform([processed_text])
        
        # Get feature names (terms)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get scores for each term
        scores = tfidf_matrix.toarray()[0]
        
        # Create a list of (term, score) tuples and sort by score
        term_scores = [(term, score) for term, score in zip(feature_names, scores) if score > 0]
        term_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out terms that are just numbers or single characters
        filtered_terms = [(term, score) for term, score in term_scores 
                         if not term.isdigit() and len(term) > 1]
        
        # Extract top keywords
        top_keywords = [term for term, score in filtered_terms[:num_keywords]]
        
        return top_keywords

if __name__ == "__main__":
    extractor = KeywordExtractor()
    jd = ("Data scientist position requiring strong machine learning experience, "
          "data analysis skills, and expertise in building scalable models "
          "for real-time data processing.")
    print("Extracted Keywords:", extractor.extract_keywords(jd, 10))