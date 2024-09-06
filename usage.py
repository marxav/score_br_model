import os
import json
import requests

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


def update_model_price_list(verbose=False):

    try:
        # get all directories in './env/lib'
        directories = [x[0] for x in os.walk('./env/lib')]

        # get the sub-directories name starting with 'python' in './env/lib'
        # limit the search to the direct sub-directories of './env/lib'
        python_dirs = [d for d in directories if 'python' in d and len(d.split('/')) == 4]
        
        # select the latest version of python in python_dirs
        python_dir = sorted(python_dirs)[-1]

        
        site_pack_dir = os.path.join(python_dir, 'site-packages')
        
        litellm_dir = os.path.join(site_pack_dir, 'litellm')
        # check that python_dir contains an litellm folder
        if 'litellm' not in os.listdir(site_pack_dir):
            print('creating: litellm folder', site_pack_dir)
            os.mkdir(litellm_dir)
        
        local_filename = 'model_prices_and_context_window_backup.json'
        llm_file = os.path.join(litellm_dir, local_filename)
        
        if verbose:
            print('python_dirs:', python_dirs)
            print('python_dir:', python_dir)
            print('site_pack_dir:', site_pack_dir)
            print('litellm_dir:', litellm_dir)
            print('llm_file:', llm_file)

        # download the latest version thee litellm file descring llms, ie file
        # https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json
        # and save it in the env/lib/python3.8/site-packages/litellm folder
        url = 'https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json'
        r = requests.get(url)
        # read byte content of the file
        new_content = r.content

        # check that litellm folder contains a file named model_prices_and_context_window.json
        if local_filename in os.listdir(litellm_dir):
            # read the content of the local file model_prices_and_context_window.json
            old_content = open(llm_file, 'rb').read()
        else:
            print('WARNING: model_prices_and_context_window.json not found in', litellm_dir)
            old_content = b''

        if verbose: 
            print('old_content:', old_content[0:200])
            print('new_content:', new_content[0:200])
        
        if old_content != new_content:
            with open(llm_file, 'wb+') as f:
                f.write(new_content)
            print('model price file updated')
        else:
            print('model price file already up-to-date')
    except Exception as e:
        print('WARNING: error while updating model price file:', e)
    return
