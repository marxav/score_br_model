# read OPENAI_API_KEY from  .env file in current directory
open_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('OPENAI_API_KEY')), None)


from datetime import datetime
import os
import sys
from openai import OpenAI
import pandas as pd
import numpy as np
import scores

def get_translation(client, model, text_br):
  response = client.chat.completions.create(
      model=model,
      messages=[{"role": "user", "content": f"Translate the following Breton text to French: {text_br}"}],
      stream=False,
      temperature=0.0,
      top_p=0.95,
  )

  text_fr_predicted = response.choices[0].message.content
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
          
  return text_fr_predicted, total_tokens, price

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
   'datetime': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
   'log_file_postfix': 'logs.tsv',
   'res_file_postix': 'res.tsv',
   'input_file': 'samples.tsv'
}

def get_data(config):
    input_file = config['input_file']
    if input_file.endswith('_br.txt'):
        br_filename = input_file
        fr_filename = br_filename[:-7] + '_fr.txt'

        if not os.path.isfile(br_filename):
            print(f'breton input file {br_filename} not found') 
            exit(-1)   
        if not os.path.isfile(fr_filename):
            print(f'french input file {fr_filename} not found') 
            exit(-1)
        # Open the file in read mode
        with open(br_filename, 'r') as file:
            text_br = file.read()
        with open(fr_filename, 'r') as file:
            text_fr_target = file.read()
    else:
        if not os.path.isfile(input_file):
            print(f'warning: did not found the tsv input file')
        
        text_br = ''
        text_fr_target = ''
        config['input_file']
        df = pd.read_csv(input_file, sep='\t')
        for index, row in df.iterrows():
            # Example texts
            print(row)
            text_br = text_br + row['Brezhoneg']
            text_fr_target = text_fr_target + row['Français']
        
    return text_br, text_fr_target


def test_model(config, translation_model, text_br, text_fr_target, verbose=True):
  # perform the translation
  text_fr_predicted, tokens, price = get_translation(config['client'], translation_model, text_br)
  if verbose:
     print('text_fr_predicted:', text_fr_predicted)

  sentences_br = text_br.split('.')
  sentences_fr_p = text_fr_predicted.split('.')
  sentences_fr_t = text_fr_target.split('.')
  
  assert len(sentences_br) == len(sentences_fr_t)
  assert len(sentences_fr_p) == len(sentences_fr_t)

  df_results = pd.DataFrame()
  for sentence_br, sentence_fr_t, sentence_fr_p in zip(sentences_br, sentences_fr_t, sentences_fr_p):

    # calculate number of words in br sentence (just for info)
    br_words = len(sentence_br.split(' '))
    if sentence_fr_t == '':
      break
    score = scores.get_camembert_score(sentence_fr_t, sentence_fr_p)

    sample_log = {
        'br': sentence_br,
        'fr_target': sentence_fr_t,
        'fr_'+translation_model: sentence_fr_p,
        'score_'+translation_model: score,
        'price': price,
        'tokens': tokens,
        'br_words': br_words
    }
    print(f'sample_log:{sample_log}')
    df_sample = pd.DataFrame([sample_log])

    # Concatenate the new row DataFrame with the existing DataFrame
    df_results = pd.concat([df_results, df_sample], ignore_index=True)
    print('df_results:', df_results)
  return df_results


def test_models(config, verbose=True):

    config['client'] = OpenAI(api_key=open_api_key)
    
    text_br, text_fr_target = get_data(config)
    if verbose:
        print('========================')
        print('text_br:', text_br)
        print('text_fr_target:', text_fr_target)
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
        df_details = test_model(config, translation_model, text_br, text_fr_target)
        if df_detailss.shape[0] == 0:
            df_detailss = df_details
        else:
            df_detailss = pd.merge(left=df_detailss, right=df_details, on=['br', 'fr_target', 'br_words'])
        score_column = 'score_'+translation_model
        score_mean = df_details[score_column].mean()
        score_std = df_details[score_column].std()
        price = df_details['price'].sum()
        tokens = df_details['tokens'].sum()
        br_words_mean = df_details['br_words'].mean()
        br_words_std = df_details['br_words'].std()
        
        result = {
            'datetime': config['datetime'],
            'model': translation_model,
            'score_mean': int(score_mean*100)/100,
            'score_std': int(score_std*100)/100,
            'price': int(price*100)/100,
            'tokens': tokens,
            'br_words_mean': int(br_words_mean*10)/10,
            'br_words_std': int(br_words_std*10)/10
        }
        print(result)
        df_result = pd.DataFrame([result])
        df_results = pd.concat([df_results, df_result], ignore_index=True)

    # write logs in a tsv file
    # put 'fr_target' as the 2nd column in the tsv file
    #columns = list(df_detailss.columns)
    #columns.remove('fr_target')
    #columns.insert(1, 'fr_target')
    #df_detailss = df_detailss[columns]
    log_filename = config['datetime'] + '_' + label + '_' + config['log_file_postfix']
    df_detailss.to_csv(log_filename, index=False, sep='\t')

    # write global results in a tsv file
    res_filename = config['datetime'] + '_' + label + '_' + config['res_file_postix']
    df_results.to_csv(res_filename, index=False, sep='\t')

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