import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from nltk.translate.meteor_score import meteor_score
from nltk.stem.snowball import FrenchStemmer
from transformers import CamembertTokenizer, CamembertModel
from torch.nn.functional import cosine_similarity
import torch

# Download the Punkt tokenizer models (if not already downloaded)
nltk.download('punkt')
nltk.download('wordnet')

def get_bleu_score(reference, candidate, language='french'):
    # Tokenize the French sentences
    reference_tokens = word_tokenize(reference, language=language)
    candidate_tokens = word_tokenize(candidate, language=language)

    # Calculate BLEU score
    score = sentence_bleu([reference_tokens], candidate_tokens)

    print(f"BLEU score: {score:.4f}")
    return score

def get_meteor_score(reference, candidate):
    # Initialize French stemmer
    stemmer = FrenchStemmer()

    # Stemming words (simple demonstration)
    reference_stemmed = [stemmer.stem(word) for word in reference]
    candidate_stemmed = [stemmer.stem(word) for word in candidate]

    # Calculate METEOR score using stemmed words
    score = meteor_score([reference_stemmed], candidate_stemmed)

    print(f"METEOR score: {score:.4f}")
    return score

# Example reference and candidate sentences in French
reference = "Le renard brun rapide a sauté par-dessus le chien énergique"
candidate = "Le renard brun rapide a sauté par-dessus le chien paresseux"
#get_bleu_score(reference, candidate)
#get_meteor_score(reference, candidate)


tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertModel.from_pretrained('camembert-base')

def text_to_embedding(text, model, tokenizer):
    # Encode the text
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Pool the outputs into a single mean vector
    embeddings = model_output.last_hidden_state.mean(dim=1)
    return embeddings

def get_camembert_score(text1, text2):
    # Convert texts to embeddings
    embedding1 = text_to_embedding(text1, model, tokenizer)
    embedding2 = text_to_embedding(text2, model, tokenizer)
    # Calculate cosine similarity using PyTorch
    similarity = cosine_similarity(embedding1, embedding2)
    return similarity.item()  # Convert tensor to a single scalar