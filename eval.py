import os
import sys
from datetime import datetime
from openai import OpenAI
import pandas as pd
import numpy as np
import scores
import input_file

# the possible translation_models, so far:
# 'gpt-3.5-turbo-0613',  # release date: 2023-06-13
# 'gpt-3.5-turbo-1106',  # release date: 2023-11-06
# 'gpt-3.5-turbo-0125',  # release date: 2024-01-25
# 'gpt-3.5-turbo',       # release date: 2024-04-09', 
# 'gpt-4-0613',         # release date: 2023-06-13 aka 'gpt-4'
# 'gpt-4-1106-preview', # release date: 2023-11-06
# 'gpt-4-0125-preview', # release date: 2024-05-25
# 'gpt-4-turbo',         # release date: 2024-04-09s aka 'gpt-4-turbo'
config = {
    'translation_models': ['gpt-3.5-turbo', 'gpt-4-turbo'],
    'tasks': ['br2fr', 'fr2br'],
    #'translation_models': ['gpt-3.5-turbo'],
    #'tasks': ['fr2br'],
    'datetime': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    'log_file_postfix': 'logs.tsv',
    'res_file_postix': 'res.tsv',
    'input_file': 'samples.tsv'
}


# read OPENAI_API_KEY from  .env file in current directory
open_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('OPENAI_API_KEY')), None)


# get the source data to be translated, as well as the ideal target data
def get_data(config):
    
    input_file = config['input_file']
    lang_src = config['lang_src']
    lang_dst = config['lang_dst']
    
    if not os.path.isfile(input_file):
        print(f'warning: did not found the tsv input file')
    text_src = ''
    text_dst_target = ''
    
    df = pd.read_csv(input_file, sep='\t', encoding = 'utf8')
    for index, row in df.iterrows():
        # Example texts
        text_src = text_src + row[lang_src]
        text_dst_target = text_dst_target + row[lang_dst]
        
    # postprocessing
    # remove '...' that will otherwise be undestood as 3 different sentences
    text_src = text_src.replace('...', '…')
    text_dst_target = text_dst_target.replace('...', '…')
    
    return config, text_src, text_dst_target


# perform the translation of a source text thanks to a given model (a.k.a. LLM)
def get_translation(config, model, text_src, verbose=False):

  lang_src = config['lang_src']
  lang_dst = config['lang_dst']
  if lang_src == 'br' and lang_dst == 'fr':
    prompt = "Translate the following Breton text to French. "
  elif lang_src == 'fr' and lang_dst == 'br':
    prompt = "Translate the following French text to Breton. "
  prompt += " Immediatly write the translated text, nothing more."
  prompt += " The translated text must contain the same number of sentences and same number of '.' characters as in the input text."
  #prompt += " If there is no '?' in the text to be translated, there must be no '?' as well in the translated text." # does not work

  client = OpenAI(api_key=open_api_key)

  response = client.chat.completions.create(
      model=model,
      messages=[{"role": "system", "content": prompt}, {"role": "user", "content": text_src}],
      stream=False,
      temperature=0.0,
      top_p=0.95,
  )

  text_dst_predicted = response.choices[0].message.content
  
  if verbose:
    print('text_dst_predicted:', text_dst_predicted)
    print(response.to_json)

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


# launch the translation with a given model and estimate (i.e. score) the result
def test_model(config, task, translation_model, text_src, text_dst_target, verbose=False):
  
  # preprocessing the input (text_src)
  text_src = text_src.rstrip().replace('\n', '')

  # perform the translation
  text_fr_predicted, tokens, price = get_translation(config, translation_model, text_src)

  is_translation_ok = False
  n_max = 3
  n = 0

  # here, we need a loop, because sometimes the translation ends up with more sentences in
  # the translated text than in the source text; hence we allow n_max trials be
  while is_translation_ok == False and n <= n_max:
    n += 1

    # postprocessing the output (text_fr_predicted) for corner cases (e.g. two consecutive points)
    text_fr_predicted = text_fr_predicted.replace('...', '…')
    text_fr_predicted = text_fr_predicted.replace('?', '?.') # hack because sometime affirmative sentence is predicted as an interrogative sentence

    if verbose:
        print('n:', n)
        print('text_src:', text_src)
        print('text_dst_target:', text_dst_target)
        print('text_fr_predicted:', text_fr_predicted)

    sentences_src = text_src.split('.')
    sentences_dst_t = text_dst_target.split('.')
    sentences_dst_p = text_fr_predicted.split('.')
    
    if verbose:
        print('sentences_src:', sentences_src)
        print('sentences_dst_t:', sentences_dst_t)
        print('sentences_dst_p:', sentences_dst_p)

    l1 = len(sentences_src)
    l2 = len(sentences_dst_t)
    l3 = len(sentences_dst_p)
    
    if verbose:
        print("++++++++++++++++++")
        print("src:"+text_src+" , n_sentences:", l1)
        print("++++++++++++++++++")
        print("dst_t:"+text_dst_target+" , n_sentences:", l2)
        print("++++++++++++++++++")
        print("dst_p:"+text_fr_predicted+" , n_sentences:", l3)
        print("++++++++++++++++++")
    
    if (l1 == l2) and (l2 == l3):
        is_translation_ok = True
    else:    
        is_translation_ok = False
        if l1 != l2:
            print('!!!warning l1:'+l1+' is different from l2:'+l2)
        else:
            print(f'!!!warning l1:{l1} is different from l3:{l3}')
        if n == n_max:
            print("ERROR, too many unsuccessful trials, let's stop here")
            exit(-1)
  
  
  df_results = pd.DataFrame()
  for sentence_src, sentence_dst_t, sentence_dst_p in zip(sentences_src, sentences_dst_t, sentences_dst_p):

    # calculate number of words in br sentence (just for info)
    br_words = len(sentence_src.split(' '))
    if sentence_dst_t == '':
      break
    score = scores.get_openai_score(sentence_dst_t, sentence_dst_p)

    sample_log = {
        'task': task,
        'datetime': config['datetime'],
        'model': translation_model,
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

    # only print the new line in the df (i.e. last line of the df)
    print('df_results:', df_results.tail(1))

  return df_results

# launch all the asks with each of supported models
def test_models(config, args, verbose=False):

    config['input_file'] =  input_file.check_input_file(args)

    df_full_results = pd.DataFrame()
    df_full_detailss = pd.DataFrame()

    # test models for each of the task listed in the config
    for task in config['tasks']: 
        print(f'* starting task {task}:')
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

        # test all the translation models listed is the config
        for translation_model in config['translation_models']:            
            print(f'  * starting translation_model {translation_model}:')
            df_details = test_model(config, task, translation_model, text_src, text_dst_target)
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

def main(args):
    test_results = test_models(config, args)
    print(test_results)

if __name__ == '__main__':
    main(sys.argv)
