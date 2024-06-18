import os
import yaml
from itertools import zip_longest
import pandas as pd 

def read_lines_and_count(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        line_count = len(lines)
        return lines, line_count
    except FileNotFoundError:
        print(f"The file {file_path} does not exist.")
        return [], 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return [], 0
    
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
    
def print_files_with_line_numbers(file1_path, file2_path):
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
            for line_number, (line1, line2) in enumerate(zip_longest(file1, file2, fillvalue=''), start=1):
                print(f"{line_number}: {line1.strip()}")
                print(f"{line_number}: {line2.strip()}")
                print('')
    except FileNotFoundError as e:
        print(f"File not found: {e.filename}")
    except Exception as e:
        print(f"An error occurred: {e}")

def create_merged_file(input_file, verbose=False):
    
    l1_filename = input_file
    if l1_filename.endswith('_br.txt'):
        l2_filename = input_file[:-7] + '_fr.txt'
        br_file = l1_filename
    elif l1_filename.endswith('_fr.txt'):        
        l2_filename = input_file[:-7] + '_br.txt'
        br_file = l2_filename
    else:
        print(f'ERROR: input_file {input_file}: unexpected format')
        exit(-1)

    if not os.path.isfile(l1_filename):
        print(f'input file l1 {l1_filename} not found') 
        exit(-1)   
    if not os.path.isfile(l2_filename):
        print(f'input file l2 {l2_filename} not found') 
        exit(-1)

    if count_lines_in_file(l1_filename) != count_lines_in_file(l2_filename):
        print_files_with_line_numbers(l1_filename, l2_filename)

    '''
    # Open each of the two files in read mode and extract their text
    with open(l1_filename, 'r') as file:
        l1_text = file.read()
    with open(l2_filename, 'r') as file:
        l2_text = file.read()


    # split the src and dst texts over multiple lines (one sentence per line)
    l1_text = l1_text.rstrip().replace('.', '.\n')
    if l1_text[-1]=='\n':
        l1_text = l1_text[:-1]
    l1_lines = l1_text.split('\n')
    l2_text = l2_text.rstrip().replace('.', '.\n')
    if l2_text[-1]=='\n':
        l2_text = l2_text[:-1]
    l2_lines = l2_text.split('\n')
    '''
    '''
    if verbose:
        print('input_file:', input_file)
        print('l1_lines:', l1_lines)
        print('l2_lines:', l2_lines)
        print('len(l1_lines):', len(l1_lines))
        print('len(l2_lines):', len(l2_lines))
    '''
    '''
    # check that both files contains the same number of sentences
    if len(l1_lines) == len(l2_lines):
        print('br and fr files do not have the same number of sentences', len(l1_lines), len(l2_lines))
        exit(-1)
    '''

    l1_lines, n1 = read_lines_and_count(l1_filename)
    l2_lines, n2 = read_lines_and_count(l2_filename)
    
    if n1 != n2:
        print(f'ERROR: n1:{n1} and n2:{n2} are different, check your input files')
        exit(-1)
    # write both text in a new tsv file
    new_input_file = input_file[:-7]+'.tsv'  

    '''     
    with open(new_input_file, 'w+') as new_file:
        new_file.write('br'+'\t'+'fr'+'\n')
        if br_file == l1_filename:        
            for l1, l2 in zip (l1_lines, l2_lines):
                new_file.write(l1.rstrip('\n') + '\t' + l2.rstrip('\n') + '\n')
                print(l1.rstrip())
                print(l2.rstrip())
                print('')
        else:
            for l2, l1 in zip (l2_lines, l1_lines):
                new_file.write(l2.rstrip('\n') + '\t' + l1.rstrip('\n') + '\n')
        new_file.close()
    '''

    if br_file == l1_filename:
        df = pd.DataFrame({
            'br': l1_lines,
            'fr': l2_lines
        })
    else:
         df = pd.DataFrame({
            'br': l2_lines,
            'fr': l1_lines
        })       

    # Write the DataFrame to a CSV file with tab separation
    df.to_csv(new_input_file, sep='\t', index=False, encoding='utf-8')

    return new_input_file

def check_dataset_file(input_file): 
    
    if not os.path.exists(input_file):
        print(f'ERROR: input file {input_file} does not exist.')
        exit(-1)
    elif input_file.endswith('.tsv'):
        return input_file
    elif input_file.endswith('_br.txt') or input_file.endswith('_fr.txt'):
        return create_merged_file(input_file)
    else:
        print('input file type is not supported')
        exit(-1)

class TaskConfig:
    def __init__(self, name, prompt, separator):
        self.name = name
        self.prompt = prompt
        self.separator = separator
 
class Config:
    def __init__(self, models, tasks, dataset_file, log_file_postfix, res_file_postix, temperature, top_p):
        self.models = models
        self.tasks = tasks
        self.dataset_file = dataset_file
        self.log_file_postfix = log_file_postfix
        self.res_file_postix = res_file_postix
        self.temperature = temperature
        self.top_p = top_p

def load_config(args): 
    # minimum check of the input file extension
    if len(args) < 2:
        print('ERROR: first argument requires an .yaml configuration file.')
        exit(-1)
    config_file = args[1] 
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
        dataset_file = directory + os.sep + config_data['dataset_file']
        log_file_postfix = config_data['log_file_postfix']
        res_file_postfix = config_data['res_file_postfix']
        temperature = config_data['temperature']
        top_p = config_data['top_p']

        config = Config(models=models, tasks=tasks, 
                        dataset_file=dataset_file,
                        log_file_postfix=log_file_postfix, res_file_postix=res_file_postfix, 
                        temperature=temperature, top_p=top_p)
        config.dataset_file =  check_dataset_file(config.dataset_file)
        return config