from transformers import CamembertTokenizer, CamembertModel
from torch.nn.functional import cosine_similarity
import torch

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
