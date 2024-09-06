import usage
from mistralai import Mistral, UserMessage

# read MISTRAL_API_KEY for Mistral models
mistral_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('MISTRAL_API_KEY')), None)

def completion(config, model, prompt, text_src, text_dst_target, verbose=False):
    
    client = Mistral(api_key=mistral_api_key)

    response = client.chat.complete(
        model=model,
        messages=[{"role":"user", "content":prompt+text_src}],
        temperature = config.temperature,
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
    price = usage.get_price(config, model, input_tokens=in_tokens, output_tokens=out_tokens)
    error=False

    return text_dst_predicted, total_tokens, price, error
