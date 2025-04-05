import re
import string

# Job-specific stopwords
JOB_STOPWORDS = {
    'job', 'position', 'work', 'working', 'company', 'candidate', 'candidates',
    'experience', 'role', 'team', 'requirements', 'required', 'skills', 'ability',
    'looking', 'qualified', 'must', 'responsibilities', 'responsible', 
    'opportunity', 'opportunities', 'duties', 'duty', 'please', 'apply',
    'email', 'resume', 'cover', 'letter', 'salary', 'applicant', 'applicants'
}

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove digits
    text = re.sub(r'\d+', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def preprocess(text):
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Remove job-specific stopwords
    words = cleaned_text.split()
    filtered_words = [word for word in words if word not in JOB_STOPWORDS and len(word) > 1]
    
    return ' '.join(filtered_words)