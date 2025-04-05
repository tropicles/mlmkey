from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
import os

def train(data_file=None):
    """
    Train the TF-IDF vectorizer on a dataset of job descriptions.
    If no data_file is provided, a model will be created but not trained.
    The model will be trained on the first job description it processes.
    """
    if data_file and os.path.exists(data_file):
        # Load job descriptions from file
        print(f"Loading data from {data_file}")
        try:
            if data_file.endswith('.csv'):
                df = pd.read_csv(data_file)
                job_descriptions = df['job_description'].tolist()  # Assumes column name is 'job_description'
            elif data_file.endswith('.txt'):
                with open(data_file, 'r', encoding='utf-8') as f:
                    job_descriptions = f.readlines()
            else:
                print(f"Unsupported file format: {data_file}")
                return
                
            # Create and fit vectorizer
            vectorizer = TfidfVectorizer(
                ngram_range=(1, 3),
                max_features=10000,
                stop_words='english'
            )
            vectorizer.fit(job_descriptions)
            
            # Save the model
            joblib.dump(vectorizer, "tfidf_model.pkl")
            print(f"Model trained on {len(job_descriptions)} documents and saved to tfidf_model.pkl")
            
        except Exception as e:
            print(f"Error training model: {e}")
    else:
        # Create a default vectorizer to be fit during first prediction
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            stop_words='english'
        )
        joblib.dump(vectorizer, "tfidf_model.pkl")
        print("Default TF-IDF vectorizer created (not trained) and saved to tfidf_model.pkl")
        print("It will be trained on the first job description it processes.")

if __name__ == "__main__":
    # You can specify a dataset file if available
    train()  # Or train("job_descriptions.csv")