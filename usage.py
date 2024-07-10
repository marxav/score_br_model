import json

def get_price(config, model, input_tokens=0, output_tokens=0):
    # read the dictionary containing the price of the models
    with open(config.models_price_file) as f:
        model_prices = json.load(f)

    # some models are provided via multiple party providers, so we need to adjust the model name
    # to find the correct price in the model prices"dictionary
    if 'llama' in model:
        model_price_name = 'groq/' + model
    elif 'mistral' in model:
        model_price_name = 'mistral/' + model
    elif 'palm' in model:
        model_price_name = 'openrouter/google/' + model
    else:
        model_price_name = model

    # get the input_cost_per_token and output_cost_per_token for the model
    # check that key model_price_name exists in the model_prices dictionary
    if model_price_name not in model_prices:
        # this can happen for a new model just released few hours ago
        # or/and if llmlite not yet updated
        print(f'ERROR: model {model_price_name} is not in the model prices file')
        input_cost_per_token = 0
        output_cost_per_token = 0
    else:
        input_cost_per_token = model_prices[model_price_name]['input_cost_per_token']
        output_cost_per_token = model_prices[model_price_name]['output_cost_per_token']

    price = input_cost_per_token * input_tokens + output_cost_per_token * output_tokens
 
    return price