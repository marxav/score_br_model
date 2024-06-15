import cohere

# read COHERE_API_KEY for Commmand-r models
cohere_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('COHERE_API_KEY')), None)

def process(config, model, prompt, text_src, text_dst_target, verbose=False):
    
    co = cohere.Client(api_key=cohere_api_key)

    response = co.chat(
        model="command-r-plus",
        message=prompt+text_src,
        temperature = config['temperature'],
        p = config['top_p'],
    )
    print('response:', response)

    try: 
        text_dst_predicted = response.text
        if verbose:
            print('text_dst_target:', text_dst_target)
            print("text predicted:", text_dst_predicted)
    except:
        print('WARNING: no response provided by the LLM. LLM response was:', response)
        error = True
        return 'N/A', 0, 0, error
    price = 0
    token_info = response.meta
    if verbose: 
        print('token_info:', token_info)
    in_tokens = token_info.billed_units.input_tokens
    out_tokens = token_info.billed_units.output_tokens
    total_tokens = in_tokens + out_tokens

    return text_dst_predicted, total_tokens, price, error