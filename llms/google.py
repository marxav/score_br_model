import google.generativeai as genai

# read GOOGLE_API_KEY for Gemini models
google_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('GOOGLE_API_KEY')), None)

def process(config, model, prompt, text_src, text_dst_target, verbose=False):
    genai.configure(api_key=google_api_key)
    google_model = genai.GenerativeModel(model)
    response = google_model.generate_content(
        prompt + text_src,
        generation_config=genai.GenerationConfig(temperature=config['temperature'], top_p=config['top_p'])
    )
    #if verbose:
    
    try:
        text_dst_predicted = response.text
    except:
        print('WARNING: no response provided by the LLM. LLM response was:', response)
        error = True
        return 'N/A', 0, 0, error
    if verbose:
        print('text_dst_target:', text_dst_target)
        print('text_dst_predicted:', text_dst_predicted)
    

    other_data = response.usage_metadata
    total_tokens = other_data.total_token_count
    in_tokens = other_data.prompt_token_count
    out_tokens = other_data.candidates_token_count
    price = 0
    error = False
    return text_dst_predicted, total_tokens, price, error