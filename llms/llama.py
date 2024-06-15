from groq import Groq

# read GROQ_API_KEY for Meta LLama models
groq_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('GROQ_API_KEY')), None)

def process(config, model, prompt, text_src, text_dst_target, verbose=False):
    
    client = Groq(api_key=groq_api_key)
    error = False
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt + text_src
            }
        ],
        model=model,
        temperature = config['temperature'],
        top_p=config['top_p']
    ) 
    if verbose:
        print('text_dst_target:', text_dst_target)
        print(response)
    try:
        text_dst_predicted = response.choices[0].message.content
    except:
        print('WARNING: no response provided by the LLM. LLM response was:', response)
        error = True
        return 'N/A', 0, 0, error
    price = 0
    in_tokens = response.usage.prompt_tokens
    out_tokens = response.usage.completion_tokens
    total_tokens = response.usage.total_tokens


    return text_dst_predicted, in_tokens, out_tokens, error