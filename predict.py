from keybert import KeyBERT
from preprocess import preprocess

class KeywordExtractor:
    def __init__(self):
        # Use a lighter model for lower resource usage
        self.model = KeyBERT("paraphrase-MiniLM-L6-v2")
    
    def extract_keywords(self, job_description, num_keywords=15):
        processed_text = preprocess(job_description)
        keywords = self.model.extract_keywords(
            processed_text, 
            keyphrase_ngram_range=(1, 3),
            stop_words="english",
            top_n=num_keywords,
            use_mmr=True,
            diversity=0.7,
            nr_candidates=50
        )
        return [kw for kw, score in keywords]

if __name__ == "__main__":
    extractor = KeywordExtractor()
    jd = ("Data scientist position requiring strong machine learning experience, "
          "data analysis skills, and expertise in building scalable models "
          "for real-time data processing.")
    print("Extracted Keywords:", extractor.extract_keywords(jd, 10))
