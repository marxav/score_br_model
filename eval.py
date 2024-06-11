# read OPENAI_API_KEY from  .env file in current directory
open_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('OPENAI_API_KEY')), None)


from datetime import datetime
import os
import sys
from openai import OpenAI
import pandas as pd
import numpy as np
import scores

def get_translation(client, model, text_src):

  lang_src = config['lang_src']
  lang_dst = config['lang_dst']
  if lang_src == 'br' and lang_dst == 'fr':
      prompt = f"Translate the following Breton text to French: {text_src}."
  elif lang_src == 'fr' and lang_dst == 'br':
      prompt = f"Translate the following French text to Breton: {text_src}."
  #prompt += "Immediatly start the translation with preamble in English."
      
  response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": prompt}],
      stream=False,
      temperature=0.0,
      top_p=0.95,
  )

  text_dst_predicted = response.choices[0].message.content
  
  #print('text_dst_predicted:', text_dst_predicted)
  #print(response.to_json)

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
          
  return text_dst_predicted, total_tokens, price

translation_models = [
    #'gpt-3.5-turbo-0613',  # release date: 2023-06-13
    #'gpt-3.5-turbo-1106',  # release date: 2023-11-06
    #'gpt-3.5-turbo-0125',  # release date: 2024-01-25
    'gpt-3.5-turbo',       # release date: 2024-04-09', 
    #'gpt-4-0613',         # release date: 2023-06-13 aka 'gpt-4'
    #'gpt-4-1106-preview', # release date: 2023-11-06
    #'gpt-4-0125-preview', # release date: 2024-05-25
    'gpt-4-turbo-2024-04-09', # release date: 2024-04-09s aka 'gpt-4-turbo'
    ]


config = {
   'translation_models': translation_models,
   'tasks': ['br2fr', 'fr2br'],
   #'tasks': ['fr2br'],
   'datetime': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
   'log_file_postfix': 'logs.tsv',
   'res_file_postix': 'res.tsv',
   'input_file': 'samples.tsv'
}

def get_data(config):
    input_file = config['input_file']
    lang_src = config['lang_src']
    lang_dst = config['lang_dst']
    if input_file.endswith('_br.txt'):
        src_filename = input_file
        dst_filename = src_filename[:-7] + '_fr.txt'

        if not os.path.isfile(src_filename):
            print(f'breton input file {src_filename} not found') 
            exit(-1)   
        if not os.path.isfile(dst_filename):
            print(f'french input file {dst_filename} not found') 
            exit(-1)
        # Open the file in read mode
        with open(src_filename, 'r') as file:
            text_br = file.read()
        with open(dst_filename, 'r') as file:
            text_fr = file.read()

        input_file = src_filename[:-7]+'.tsv'
        config['input_file'] = input_file
        with open(input_file, 'w+') as new_file:
            new_file.write('br'+'\t'+'fr'+'\n')
            new_file.write(text_br.strip()+'\t'+text_fr.strip()+'\n')
            new_file.close()
    
    if not os.path.isfile(input_file):
        print(f'warning: did not found the tsv input file')
    text_src = ''
    text_dst_target = ''
    config['input_file']
    df = pd.read_csv(input_file, sep='\t')
    for index, row in df.iterrows():
        # Example texts
        print(row)
        text_src = text_src + row[lang_src]
        text_dst_target = text_dst_target + row[lang_dst]
    
    return config, text_src, text_dst_target


def test_model(config, translation_model, text_src, text_dst_target, verbose=True):
  # perform the translation
  text_fr_predicted, tokens, price = get_translation(config['client'], translation_model, text_src)
  if verbose:
     print('text_fr_predicted:', text_fr_predicted)

  sentences_src = text_src.split('.')
  sentences_dst_p = text_fr_predicted.split('.')
  sentences_dst_t = text_dst_target.split('.')
  
  assert len(sentences_src) == len(sentences_dst_t)
  assert len(sentences_dst_p) == len(sentences_dst_t)

  df_results = pd.DataFrame()
  for sentence_src, sentence_dst_t, sentence_dst_p in zip(sentences_src, sentences_dst_t, sentences_dst_p):

    # calculate number of words in br sentence (just for info)
    br_words = len(sentence_src.split(' '))
    if sentence_dst_t == '':
      break
    score = scores.get_openai_score(sentence_dst_t, sentence_dst_p)

    sample_log = {
        'src': sentence_src,
        'dst_target': sentence_dst_t,
        'dst_'+translation_model: sentence_dst_p,
        'score_'+translation_model: score,
        'price': price,
        'n_tokens': tokens,
        'src_n_words': br_words
    }
    print(f'sample_log:{sample_log}')
    df_sample = pd.DataFrame([sample_log])

    # Concatenate the new row DataFrame with the existing DataFrame
    df_results = pd.concat([df_results, df_sample], ignore_index=True)
    print('df_results:', df_results)
  return df_results


def test_models(config, verbose=True):

    config['client'] = OpenAI(api_key=open_api_key)

    df_full_results = pd.DataFrame()
    df_full_detailss = pd.DataFrame()

    for task in config['tasks']: 

        if task == 'br2fr':
            config['lang_src'] = 'br'
            config['lang_dst'] = 'fr'
        elif task == 'fr2br':
            config['lang_src'] = 'fr'
            config['lang_dst'] = 'br'
        else:
            print(f'Error: task {task} not implemented!')
            exit(-1)
              
        config, text_src, text_dst_target = get_data(config)

        if verbose:
            print('========================')
            print('text_src:', text_src)
            print('text_dst_target:', text_dst_target)
            print('------------------------')  

        # prepare label for logs and results
        if config['input_file'].endswith('.tsv'):
            label = config['input_file'].replace('.tsv', '')
        elif config['input_file'].endswith('.txt'):
            label = config['input_file'].replace('.txt', '')
        else:
            label = config['input_file']
            print(f"warning: input_file {config['input_file']} does not finish with .tsv or .txt")
                

        df_results = pd.DataFrame()
        df_detailss = pd.DataFrame()

        for translation_model in config['translation_models']:
            df_details = test_model(config, translation_model, text_src, text_dst_target)
            if df_detailss.shape[0] == 0:
                df_detailss = df_details
            else:
                on_columns = ['src', 'dst_target', 'src_n_words']
                df_detailss = pd.merge(left=df_detailss, right=df_details, on=on_columns)
            score_column = 'score_'+translation_model
            score_mean = df_details[score_column].mean()
            score_std = df_details[score_column].std()
            price = df_details['price'].sum()
            tokens = df_details['n_tokens'].sum()
            src_n_words_mean = df_details['src_n_words'].mean()
            src_n_words_std = df_details['src_n_words'].std()
            
            result = {
                'task': task,
                'datetime': config['datetime'],
                'model': translation_model,
                'score_mean': int(score_mean*100)/100,
                'score_std': int(score_std*100)/100,
                'price': int(price*100)/100,
                'n_tokens': tokens,
                'src_n_words_mean': int(src_n_words_mean*10)/10,
                'src_n_words_std': int(src_n_words_std*10)/10
            }
            print(result)
            df_result = pd.DataFrame([result])
            df_results = pd.concat([df_results, df_result], ignore_index=True)

        df_full_detailss = pd.concat([df_full_detailss, df_detailss], ignore_index=True)
        df_full_results = pd.concat([df_full_results, df_results], ignore_index=True)

    # write logs in a tsv file
    log_filename = label + '_' + config['log_file_postfix']
    df_full_detailss.to_csv(log_filename, index=False, sep='\t')
    # write logs in a seconf file which name contains the date
    log_filename = config['datetime'] + '_' + label + '_' + config['log_file_postfix']
    df_full_detailss.to_csv(log_filename, index=False, sep='\t')
        
    # write global results in a tsv file
    res_filename = label + '_' + config['res_file_postix']
    df_full_results.to_csv(res_filename, index=False, sep='\t')
    # write logs in a seconf file which name contains the date
    res_filename = config['datetime'] + '_' + label + '_' + config['res_file_postix']
    df_full_results.to_csv(res_filename, index=False, sep='\t')
        
    return df_results


#config['input_file'] = 'samples.tsv'
#config['input_file'] = 'tregor_2110_br.txt'
if len(sys.argv) < 2:
    print('first argument requires an .tsv or .txt input file.')
    exit(-1)
config['input_file'] = sys.argv[1]

# Check if path exits
if not os.path.exists(config['input_file']):
    print('input_file does not exist')
    exit(-1)

print(test_models(config).to_string)