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

def get_openai_score(config, text1, text2):
    embedding_1, total_tokens_1 = get_embedding(config, text1)
    embedding_2, total_tokens_2 = get_embedding(config, text2)
    total_tokens = total_tokens_1 + total_tokens_2
    print('total_tokens_1:', total_tokens_1)
    print('total_tokens_2:', total_tokens_2)
    price = usage.get_price(config, config.scoring_model, input_tokens=total_tokens, output_tokens=0)
    #similarity = 1 - cosine(embedding_1, embedding_2)
    similarity = np.dot(embedding_1,embedding_2)
    #print('similarity:', similarity)
    return similarity, total_tokens, price
