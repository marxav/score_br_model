import os
import sys
from datetime import datetime
import pandas as pd
import scores
import input_file
from llms import anthropic, cohere, google, llama, mistral, openai

# the possible translation_models, so far:
# From https://platform.openai.com/docs/models/
# 'gpt-3.5-turbo-0613',  # release date: 2023-06-13
# 'gpt-3.5-turbo-1106',  # release date: 2023-11-06
# 'gpt-3.5-turbo-0125',  # release date: 2024-01-2, aka 'gpt-3.5-turbo'
# 'gpt-4-0613',          # release date: 2023-06-13, aka 'gpt-4'
# 'gpt-4-1106-preview',  # release date: 2023-11-06
# 'gpt-4-0125-preview',  # release date: 2024-05-25
# 'gtp-4-turbo-2024-04-09',  # release date: 2024-04-09, aka 'gpt-4-turbo'
# 'gpt-4o-2024-05-13',   # release date: 2024-05-13, aka 'gpt-4o'
# From https://ai.google.dev/gemini-api/docs/models/gemini?hl=fr 
# gemini-1.0-pro
# gemini-1.0-pro-001
# gemini-1.0-pro-latest
# gemini-1.0-pro-vision-latest
# gemini-1.5-flash
# gemini-1.5-flash-001
# gemini-1.5-flash-latest
# gemini-1.5-pro
# gemini-1.5-pro-001
# gemini-1.5-pro-latest
# gemini-pro
# gemini-pro-vision
# From https://docs.anthropic.com/en/docs/models-overview
#'claude-3-haiku-20240307', 
# 'claude-3-sonnet-20240229', 
# 'claude-3-opus-20240229',
config = {
    'translation_models': [
        #'command-r-plus',
        #'open-mistral-7b', 'mistral-large-latest', # not (yet?) supported: 'open-mixtral-8x7b', #'open-mixtral-8x22b', 
        #'llama3-8b-8192', 'llama3-70b-8192',
        #'claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229',
        #'gemini-1.5-flash', 'gemini-1.0-pro-001', 'gemini-1.5-pro-001', 
        #'gpt-3.5-turbo-0125', 'gpt-4-0613', 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13'
        #'gpt-4-turbo-2024-04-09', 
        'gpt-4o-2024-05-13', 
        #'gemini-1.5-pro-001'
    ],
    'tasks': [
        'br2fr',
        #'fr2br'
    ],
    'log_file_postfix': 'logs.tsv',
    'res_file_postix': 'res.tsv',
    'input_file': 'samples.tsv',
    'temperature': 0.0,
    'top_p': 0.95,
}

# get the source data to be translated, as well as the ideal target data
def get_data(config, verbose=False):
    
    input_file = config['input_file']
    lang_src = config['lang_src']
    lang_dst = config['lang_dst']
    
    if not os.path.isfile(input_file):
        print(f'warning: did not found the tsv input file')
    text_src = ''
    text_dst_target = ''
    
    df = pd.read_csv(input_file, sep='\t', encoding = 'utf8')
    df = df.dropna()
    print(df)
    for index, row in df.iterrows():
        # Example texts
        if verbose:
            print(row)
            print('text_src:', text_src)
            print('txt_src:', row[lang_src])
            print('txt_dst:', row[lang_dst])
        text_src = text_src + row[lang_src]
        text_dst_target = text_dst_target + row[lang_dst]
        
    # postprocessing
    # remove '...' that will otherwise be undestood as 3 different sentences
    text_src = text_src.replace('...', '…')
    text_dst_target = text_dst_target.replace('...', '…')
    
    return config, text_src, text_dst_target


# perform the translation of a source text thanks to a given model (a.k.a. LLM)
def get_translation(config, model, text_src, text_dst_target, verbose=False):

  lang_src = config['lang_src']
  lang_dst = config['lang_dst']
  if lang_src == 'br' and lang_dst == 'fr':
    prompt = "Translate the following Breton text to French. "
  elif lang_src == 'fr' and lang_dst == 'br':
    prompt = "Translate the following French text to Breton. "
  prompt += "Immediatly write the translated text, nothing more. Do not add any personal comment beyond translation, just translate. "
  prompt += "The translated text must contain the same number of sentences and same number of '.' characters as in the input text. "
  prompt += "\n"
  #prompt += " If there is no '?' in the text to be translated, there must be no '?' as well in the translated text." # does not work

  error = False

  if 'gpt' in model:
    res = openai.process(config, model, prompt, text_src, text_dst_target)
  elif 'gemini' in model:
    res = google.process(config, model, prompt, text_src, text_dst_target)
  elif 'claude' in model:
    res = anthropic.process(config, model, prompt, text_src, text_dst_target)
  elif 'llama' in model:
    res = llama.process(config, model, prompt, text_src, text_dst_target)
  elif 'mistral' in model:
    res = mistral.process(config, model, prompt, text_src, text_dst_target)
  elif 'command-r' in model:
    res = cohere.process(config, model, prompt, text_src, text_dst_target)
  else:
      print(f'ERROR model {model} is not supported.')
      error = True
      return 'N/A', 0, 0, True
  
  (text_dst_predicted, in_tokens, out_tokens, error) = res
  total_tokens = in_tokens, out_tokens
  price = 0.0
  error = False
  return text_dst_predicted, total_tokens, price, error


# launch the translation with a given model and estimate (i.e. score) the result
def test_model(config, task, translation_model, text_src, text_dst_target, verbose=False):
  
  # preprocessing the input (text_src)
  text_src = text_src.rstrip().replace('\n', '')

  error = False

  is_translation_ok = False
  n_max = 3
  n = 0

  # here, we need a loop, because sometimes the translation ends up with more sentences in
  # the translated text than in the source text; hence we allow n_max trials be
  while is_translation_ok == False and n <= n_max:
    n += 1

    # perform the translation
    text_fr_predicted, tokens, price, error = get_translation(config, translation_model, text_src, text_dst_target)
    if error:
        error = True
        return None, error

    # postprocessing the output (text_fr_predicted) for corner cases (e.g. two consecutive points)
    text_fr_predicted = text_fr_predicted.replace('...', '…')
    #text_fr_predicted = text_fr_predicted.replace('?', '?.') # hack because sometime affirmative sentence is predicted as an interrogative sentence

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
            print('text_src:', text_src)
            print('text_dst_target:', text_dst_target)
        else:
            print(f'!!!warning l1:{l1} is different from l3:{l3}')
            print('text_src:', text_src)
            print('text_fr_predicted:', text_fr_predicted)
        if n == n_max:
            print("ERROR, too many unsuccessful trials, let's stop here")
            error = True
            return None, error
  
  
  df_results = pd.DataFrame()
  for sentence_src, sentence_dst_t, sentence_dst_p in zip(sentences_src, sentences_dst_t, sentences_dst_p):

    # calculate number of words in br sentence (just for info)
    br_words = len(sentence_src.split(' '))
    if sentence_dst_t == '':
      break
    score = scores.get_openai_score(sentence_dst_t, sentence_dst_p)

    sample_log = {
        'task': task,
        'model': translation_model,
        'src': sentence_src,
        'target': sentence_dst_t,
        'prediction': sentence_dst_p,
        'score': score,
        'price': price,
        'n_tokens': tokens,
        'src_n_words': br_words
    }
    print('sample_log:')
    for key, value in sample_log.items():
        print(f"* {key}: {value}")
    df_sample = pd.DataFrame([sample_log])

    # Concatenate the new row DataFrame with the existing DataFrame
    df_results = pd.concat([df_results, df_sample], ignore_index=True)

    # only print the new line in the df (i.e. last line of the df)
    print('df_results:', df_results.tail(1))

  return df_results, error

# launch all the asks with each of supported models
def test_models(config, args, verbose=False):

    config['input_file'] =  input_file.check_input_file(args)


    # prepare label for logs and results
    if config['input_file'].endswith('.tsv'):
        label = config['input_file'].replace('.tsv', '')
    else:
        label = config['input_file']
        print(f"warning: input_file {config['input_file']} does not finish with .tsv")
    
    log_filename = label + '_' + config['log_file_postfix']
    res_filename = label + '_' + config['res_file_postix']


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
                
        # test all the translation models listed is the config
        for translation_model in config['translation_models']:          

            # create output file if not existing, otherwise open them
            if not os.path.exists(log_filename) or not os.path.exists(res_filename):
                df_full_results = pd.DataFrame()
                df_full_detailss = pd.DataFrame()
            else:
                df_full_results = pd.read_csv(res_filename, sep='\t')
                df_full_detailss = pd.read_csv(log_filename, sep='\t')

            print(f'  * starting translation_model {translation_model}:')
            df_details, error = test_model(config, task, translation_model, text_src, text_dst_target)
            if error:
                print(f"WARNING: error encountered with model {translation_model}, let's skip it")
                continue
            
            score_mean = df_details['score'].mean()
            score_std = df_details['score'].std()
            price = df_details['price'].sum()
            tokens = df_details['n_tokens'].sum()
            src_n_words_mean = df_details['src_n_words'].mean()
            src_n_words_std = df_details['src_n_words'].std()
            
            result = {
                'task': task,
                'datetime': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
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

            df_full_detailss = pd.concat([df_full_detailss, df_details], ignore_index=True)
            df_full_results = pd.concat([df_full_results, df_result], ignore_index=True)

            # sort results before writing them to the file
            df_full_results = df_full_results.sort_values(['task', 'score_mean'], ascending = [True, False])
        
            # append global results in the tsv results file    
            df_full_results.to_csv(res_filename, index=False, sep='\t')

            # write logs in a second file which name contains the date
            #dated_res_filename = result['datetime'] + '_' + label + '_' + config['res_file_postix']
            #df_full_results.to_csv(dated_res_filename, index=False, sep='\t')

            # append logs in the tsv logs file
            df_full_detailss.to_csv(log_filename, index=False, sep='\t')
            # write logs in a second file which name contains the date
            #dated_log_filename = result['datetime'] + '_' + label + '_' + config['log_file_postfix']
            #df_details.to_csv(dated_log_filename, index=False, sep='\t')
            
    return df_full_results

def main(args):
    test_results = test_models(config, args)
    print('test_results:', test_results)

if __name__ == '__main__':
    main(sys.argv)