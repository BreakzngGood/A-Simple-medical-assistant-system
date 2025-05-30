# Nltk to process the user's input
import streamlit as st
import medical_assistant_package.load_models as lm
import medical_assistant_package.load_datasets as ld

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import string
from sentence_transformers import SentenceTransformer, util

data = ld.load_data()

# Nltk to process the user's input
def preprocess_text_nltk(text):

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))

    words = word_tokenize(text)
    
    # Remove stop words (such as "the", "is", "and")
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return " ".join(words)

# Get the prompts of symptoms by user's input
def extract_prompts_from_dataset(text):
    
    # preprocess user'input text
    text = preprocess_text_nltk(text)

    # Calculate the similarity between text and the prompt dataset
    Sentence_model = lm.load_sentence_transformer()
    text_embedding = Sentence_model.encode(text, convert_to_tensor=True)
    prompt_embeddings = Sentence_model.encode(data["prompt_list"], convert_to_tensor=True)

    # Calculate cosine similarity
    cosine_scores = util.pytorch_cos_sim(text_embedding, prompt_embeddings)[0].cpu().numpy() 
    
    # Filter out prompts with similarity > 0.5.
    similar_prompts = sorted(
        [(prompt, score) for prompt, score in zip(data["prompt_list"], cosine_scores) if score > 0.5],
        key=lambda x: x[1], reverse=True
    )
    
    if not similar_prompts:
        st.write("No symptom found")
        return None, None
    
    prob_prompt_list = [prompt for prompt, score in similar_prompts]

    return similar_prompts , prob_prompt_list