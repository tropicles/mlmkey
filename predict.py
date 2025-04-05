from keybert import KeyBERT
from preprocess import preprocess

class KeywordExtractor:
    def __init__(self):
        # Initialize KeyBERT with a transformer model
        self.model = KeyBERT('all-mpnet-base-v2')
    
    def extract_keywords(self, job_description, num_keywords=15):
        processed_text = preprocess(job_description)
        # Extract keywords using KeyBERT with enhanced parameters
        keywords = self.model.extract_keywords(
            processed_text, 
            keyphrase_ngram_range=(1, 3),  # Consider unigrams, bigrams, and trigrams
            stop_words='english',
            top_n=num_keywords,
            use_mmr=True,  # Enable Maximal Marginal Relevance
            diversity=0.7,  # Adjust diversity for keyword selection
            nr_candidates=50  # Number of candidates to consider
        )
        # Return only the keyword phrases (ignoring scores)
        return [kw for kw, score in keywords]

if __name__ == "__main__":
    extractor = KeywordExtractor()
    jd = ("Data scientist position requiring strong machine learning experience, "
          "data analysis skills, and expertise in building scalable models "
          "for real-time data processing.")
    print("Extracted Keywords:", extractor.extract_keywords(jd, 10))
