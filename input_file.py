import os
import yaml


import requests
import json

# Function to fetch JSON content from a URL and load it into a Python variable
def fetch_json_from_url(url):
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    data = response.json()  # Parse the JSON content
    return data

openrouter_models_pricing = {} 

# Function to extract model IDs from the JSON data
def extract_openrouter_supported_model_names(json_data, verbose=False):
    models = []
    for model in json_data.get('data', []):
        id = model['id']
        prompt = float(model['pricing']['prompt'])
        completion = float(model['pricing']['completion'])
        models.append(id)
        pricing = {'in': prompt, 'out': completion}
        openrouter_models_pricing[id] = pricing
    if verbose:
        print('openrouter models:', models)
    return models

def get_openrouter_price(config, model, input_tokens=0, output_tokens=0, verbose=True):

    try:
        if verbose:
            print('openrouter model:', model)
        pricing = openrouter_models_pricing[model]
        i = input_tokens * pricing['in'] 
        o = output_tokens * pricing['out']
        price = i + o 
        if verbose:
            print(f'price={i}+{o}={price}')
        return price
    except:
        print('ERROR: pricing not found in get_openrouter_price, model:', model)
        return 0.0

def get_openrouter_models(verbose=False):
    # Example usage
    url = 'https://openrouter.ai/api/v1/models'
    try:
        models_data = fetch_json_from_url(url)
    except:
        print('ERROR: fetching {url} failed')
        models_data = None
    if verbose:
        print(json.dumps(models_data, indent=2))

    models = extract_openrouter_supported_model_names(models_data)
    bl_models = [
        'meta-llama/llama-3.1-405b',
        'mistralai/codestral-mamba',
        'perplexity/llama-3-sonar-large-32k-online',
        'perplexity/llama-3-sonar-large-32k-chat',
        'perplexity/llama-3-sonar-small-32k-online',
        'perplexity/llama-3-sonar-small-32k-chat',
        'mancer/weaver',
        'openai/gpt-3.5-turbo-0301',
        'openai/gpt-3.5-turbo'
    ]
    #bl_models = []
    models2 = []
    i=0
    for model in models:
        i+=1
        if model in bl_models:
            continue
        else:
            models2.append(model)
        if i==1000:
            break
    models=models2
    print('nb of openrouter models:', len(models))
    return models

def get_openrouter_model(wanted_model):
    # Example usage
    url = 'https://openrouter.ai/api/v1/models'
    try:
        models_data = fetch_json_from_url(url)
    except:
        print('ERROR: fetching {url} failed')
        models_data = None
    
    models = extract_openrouter_supported_model_names(models_data)
    
    for model in models:
        
        if model in wanted_model:
            return model
        
    print(f'ERROR: model:{model} not found in openrouter list of avaible model')
    exit(-1)

# the function returns the path to Litellm package's pricing file installed locally 
# (i.e. in the virtual environment), which is a copy of the pricing file available at
# https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json
def get_pricing_file():

    # look for all directories in "env/lib/"
    # look for a single directory name starting with "python"
    # exit if more than one directory is found
    root_dir = './env/lib/'
    python_dirs = [d for d in os.listdir('env/lib/') if d.startswith('python')]
    if len(python_dirs) != 1:
        print(f"ERROR: found {len(python_dirs)} python directories in env/lib/: {python_dirs}")
        exit(-1)
    python_dir = python_dirs[0]
    litellm_dir = 'env/lib/' + python_dir + '/site-packages/litellm'
    filename = 'model_prices_and_context_window_backup.json'

    pricing_file = litellm_dir + os.sep + filename
    print('pricing_file:', pricing_file)
    return pricing_file

# the function reads the lines from a file and returns them as a list of strings
def read_lines_and_count(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        l=0
        for line in lines:
            # remove the newline character at the end of the line
            lines[l] = line.rstrip()
            l+=1
        line_count = len(lines)
        return lines, len(lines)
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return [], 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], 0
    
# the function counts the number of lines in a file
def count_lines_in_file(file_path):
    try:
        with open(file_path, 'r') as file:
            line_count = sum(1 for line in file)
        return line_count
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0

# the function reads the glossary file and returns the lines as a list of strings
def get_glossary_lines(input_file):
    
    filename = input_file[:-len('_br.txt')] + '_gloss.txt'
    lines, n3 = read_lines_and_count(filename)
    return lines

# check if the source and target files exist and have the same number of lines
def check_source_and_target_files(config): 
    
    if not os.path.exists(config.source_file):
        print(f'ERROR: input file {config.source_file} does not exist.')
        exit(-1)
    if not os.path.exists(config.target_file):
        print(f'WARNING: target file {config.target_file} does not exist: evaluation will not be done.')
        config.eval = False
    else:
        source_lines, source_n_lines = read_lines_and_count(config.source_file)
        target_lines, target_n_lines = read_lines_and_count(config.target_file)

        # check that number of lines in the source and target files are the same
        if source_n_lines < 2:
                print(f'ERROR: source file {config.source_file} must contain at least 2 lines.')
                exit(-1)
        if source_n_lines != target_n_lines:
                print(f'ERROR: source file {config.source_file} has {source_n_lines} lines, while target file {config.target_file} has {target_n_lines} lines.')
                exit(-1)

        if config.split_sentences_during_eval:
            # check that number of sentences in the source and target files are the same
            source_n_sentences = sum([len(line.split('.')) for line in source_lines])
            target_n_sentences = sum([len(line.split('.')) for line in target_lines])

            if source_n_sentences != target_n_sentences:
                print(f'ERROR: source file {config.source_file} has {source_n_sentences} sentences, while target file {config.target_file} has {target_n_sentences} sentences.')
                exit(-1)
            
        config.eval = True
    return config

class TaskConfig:
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = prompt
 
class Config:
    def __init__(self, models, tasks, source_file, target_file, log_file_postfix, res_file_postix, temperature, top_p, split_sentences_during_eval):
        self.models = models
        self.tasks = tasks
        self.source_file = source_file
        self.target_file = target_file
        self.log_file_postfix = log_file_postfix
        self.res_file_postix = res_file_postix
        self.temperature = temperature
        self.top_p = top_p
        self.split_sentences_during_eval = split_sentences_during_eval


# the function loads the configuration file from a yaml input file
def load_config(args, check=True): 
    # minimum check of the input file extension
    if len(args) < 2:
        print('ERROR: first argument requires an .yaml configuration file.')
        exit(-1)
    config_file = args[1] 
    if not config_file.endswith('.yaml'):
        print(f'ERROR: configuration file must be a .yaml file (received {config_file})')
        exit(-1)
    if not os.path.exists(config_file):
        print(f'ERROR: yaml file {config_file} does not exist.')
        exit(-1)
    print('yaml config file:', config_file)

    filename = os.path.basename(config_file)
    directory = os.path.dirname(config_file)
    
    openrouter_models_all = []

    with open(config_file, 'r') as file:
        
        config_data = yaml.safe_load(file)
        
        models = [model for model in config_data['models']]
        print(models)
        for model in models:
            # if '/' in the model name, then it's a model and model price from openrouter
            if '/' in model and len(openrouter_models_all) == 0:
                openrouter_models_all = get_openrouter_models()
        for model in models:
            if model == 'openrouter/all':
                # add all models available at OpenRouter
                models.remove(model)
                models = models + openrouter_models_all
            elif '/' in model and model not in openrouter_models_all:
                models.append(model)            
        tasks = [TaskConfig(**task) for task in config_data['tasks']]
        if len(directory) > 0:
            source_file = directory + os.sep + config_data['source_file']
            target_file = directory + os.sep + config_data['target_file']
        else:
            # source_file is required
            source_file = config_data['source_file']
            # target_file is optional (but required for evaluation)
            try:
                target_file = config_data['target_file']
            except:
                target_file = 'n/a'
        log_file_postfix = config_data['log_file_postfix']
        res_file_postfix = config_data['res_file_postfix']
        temperature = config_data['temperature']
        top_p = config_data['top_p']

        # config param 'split_sentences_during_eval' is optional, is True by default
        try:
            split_sentences_during_eval = config_data.get('split_sentences_during_eval')
            if split_sentences_during_eval is None:
                split_sentences_during_eval = True
        except:
            split_sentences_during_eval = True

        config = Config(models=models, tasks=tasks, 
                        source_file=source_file,
                        target_file=target_file,
                        log_file_postfix=log_file_postfix, res_file_postix=res_file_postfix, 
                        temperature=temperature, top_p=top_p, split_sentences_during_eval=split_sentences_during_eval)
        #config.dataset_file =  check_dataset_file(config.dataset_file, check)
        check_source_and_target_files(config)
        # load the optional glossary file
        config.glossary_lines = get_glossary_lines(config.source_file)
        config.models_price_file = get_pricing_file()
        config.scoring_model = 'text-embedding-3-large'
        return config
