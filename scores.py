import numpy as np
from openai import OpenAI
import usage

open_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('OPENAI_API_KEY')), None)
embedding_client = OpenAI(api_key=open_api_key)
  
def get_embedding(config, text):
    model = config.scoring_model
    text = text.replace("\n", " ")
    response = embedding_client.embeddings.create(input = [text], model=model)
    embedding = response.data[0].embedding
    total_input_tokens = response.usage.total_tokens

    return embedding, total_input_tokens

def cosine_similarity(vec1, vec2):
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return np.dot(vec1, vec2) / (norm_vec1 * norm_vec2)

def get_openai_score(config, text1, text2, verbose=False):
    embedding_1, total_tokens_1 = get_embedding(config, text1)
    embedding_2, total_tokens_2 = get_embedding(config, text2)
    total_tokens = total_tokens_1 + total_tokens_2
    if verbose:
        print('embedding_1:', embedding_1)
        print('embedding_2:', embedding_2)
        print('total_tokens_1:', total_tokens_1)
        print('total_tokens_2:', total_tokens_2)
    price = usage.get_price(config, config.scoring_model, input_tokens=total_tokens, output_tokens=0)
    similarity = np.dot(embedding_1,embedding_2)
    if verbose:
        print('dotpro_similarity:', similarity)
        # check that the cosine similarity is the same than the dot product similarity
        # it seems to be the case with the current model, but it is not always the case
        similarity_2 = cosine_similarity(embedding_1, embedding_2)
        print('cosine_similarity:', similarity_2)
    return similarity, total_tokens, price
