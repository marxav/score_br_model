import os
import yaml

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
        if source_n_lines < 2:
            print(f'ERROR: source file {config.source_file} must contain at least 2 lines.')
            exit(-1)
        if source_n_lines != target_n_lines:
            print(f'ERROR: source file {config.source_file} has {source_n_lines} lines, while target file {config.target_file} has {target_n_lines} lines.')
            exit(-1)
        config.eval = True
    return config

class TaskConfig:
    def __init__(self, name, prompt):
        self.name = name
        self.prompt = prompt
 
class Config:
    def __init__(self, models, tasks, source_file, target_file, log_file_postfix, res_file_postix, temperature, top_p):
        self.models = models
        self.tasks = tasks
        self.source_file = source_file
        self.target_file = target_file
        self.log_file_postfix = log_file_postfix
        self.res_file_postix = res_file_postix
        self.temperature = temperature
        self.top_p = top_p

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

    with open(config_file, 'r') as file:
        
        config_data = yaml.safe_load(file)
        
        models = [model for model in config_data['models']]
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

        config = Config(models=models, tasks=tasks, 
                        source_file=source_file,
                        target_file=target_file,
                        log_file_postfix=log_file_postfix, res_file_postix=res_file_postfix, 
                        temperature=temperature, top_p=top_p)
        #config.dataset_file =  check_dataset_file(config.dataset_file, check)
        check_source_and_target_files(config)
        # load the optional glossary file
        config.glossary_lines = get_glossary_lines(config.source_file)
        config.models_price_file = get_pricing_file()
        config.scoring_model = 'text-embedding-3-large'
        return config