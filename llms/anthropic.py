import usage
import anthropic

# read ANTHROPIC_API_KEY for Gemini models
anthropic_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('ANTHROPIC_API_KEY')), None)

def completion(config, model, prompt, text_src, text_dst_target, verbose=False):
    
    message = anthropic.Anthropic(api_key=anthropic_api_key).messages.create(
        model=model,
        max_tokens=1000,
        messages=[
            {"role": "user", "content": prompt + text_src},
            #{"role": "assistant", "content": prompt}
        ],
        temperature = config.temperature,
        top_p=config.top_p,
    )
    if verbose: 
        print('text_dst_target:', text_dst_target)
        print('message.content:', message)
        print('text:', message.content[0].text)

    in_tokens = message.usage.input_tokens
    out_tokens = message.usage.output_tokens
    total_tokens = in_tokens + out_tokens
    text_dst_predicted = message.content[0].text
    price = usage.get_price(config, model, input_tokens=in_tokens, output_tokens=out_tokens)
    error = False
    return text_dst_predicted, total_tokens, price, error