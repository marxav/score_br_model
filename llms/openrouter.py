import usage
from openai import OpenAI

# read OPENROUTER_API_KEY for GPT models
openrouter_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('OPENROUTER_API_KEY')), None)

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)

def completion(config, model, prompt, text_src, text_dst_target, verbose=False):
    
    response = client.chat.completions.create(
        model= model,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text_src}],
        stream=False,
        temperature = config.temperature,
        top_p=config.top_p,
    )
    print('OPENROUTER response:', response)
    text_dst_predicted = response.choices[0].message.content

    # remove leading space
    if text_dst_predicted.startswith(' '):
        text_dst_predicted = text_dst_predicted[1:]
    
    if verbose:
        print('text_dst_target:', text_dst_target)
        print('text_dst_predicted:', text_dst_predicted)
        print(response.to_json)
        
    total_tokens = response.usage.total_tokens
    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens

    price = usage.get_price(config, model, input_tokens=in_tokens, output_tokens=out_tokens)
    error = False
    
    return text_dst_predicted, total_tokens, price, error
