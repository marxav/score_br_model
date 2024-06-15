from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage

# read MISTRAL_API_KEY for Mistral models
mistral_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('MISTRAL_API_KEY')), None)

def process(config, model, prompt, text_src, text_dst_target, verbose=False):
    
    client = MistralClient(api_key=mistral_api_key)

    response = client.chat(
        model=model,
        messages=[ChatMessage(role="user", content=prompt+text_src)],
        temperature = config['temperature'],
        top_p=1.0
    )
    if verbose:
        print(response)
    try: 
        text_dst_predicted = response.choices[0].message.content
        if verbose:
            print('text_dst_target:', text_dst_target)
            print("text predicted:", text_dst_predicted)
    except:
        print('WARNING: no response provided by the LLM. LLM response was:', response)
        error = True
        return 'N/A', 0, 0, error
    price = 0
    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens

    return text_dst_predicted, in_tokens, out_tokens, error