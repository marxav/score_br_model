from openai import OpenAI

# read OPENAI_API_KEY for GPT models
openai_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('OPENAI_API_KEY')), None)

client = OpenAI(api_key=openai_api_key)

def process(config, model, prompt, text_src, text_dst_target, verbose=False):

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text_src}],
        stream=False,
        temperature = config['temperature'],
        top_p=config['top_p'],
    )

    text_dst_predicted = response.choices[0].message.content

    if verbose:
        print('text_dst_target:', text_dst_target)
        print('text_dst_predicted:', text_dst_predicted)
        print(response.to_json)

    total_tokens = response.usage.total_tokens
    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens

    price = 0
    if "3.5" in model:
        price = in_tokens *0.5/1e6 + out_tokens *1.5/1e6
    elif "4" in model:
        price = in_tokens *5/1e6 + out_tokens *15/1e6
    else:
        price = 0
        print('error: model unknown!!!')
    error = False
    
    return text_dst_predicted, total_tokens, price, error
    