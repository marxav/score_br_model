import os
import sys
from datetime import datetime
import pandas as pd
import scores
import usage
import input_file
from litellm import completion
from llms import anthropic, cohere, google, llama, mistral, openai, palm
import pandas as pd

def save_results(config, model, task, translated_lines):
    
  output_file = get_output_file(config, model, task)
  l_max = len(translated_lines)
  l=1
  with open(output_file, 'w', encoding='utf-8') as f:
    for line in translated_lines:
      if l != l_max: 
        f.write(line+'\n')
      else:
        f.write(line)
      l+=1

  print(f'results for model:{model} on task:{task.name} saved in file:{output_file}')

# get data from a file (e.g. source data file, target data file)
def get_data(config, file, verbose=False):
    
  lines, n_lines = input_file.read_lines_and_count(file)
  
  if n_lines == 0:
    print(f'ERROR: no data found in {file}')
    exit(-1)
  if verbose:
    print(f'get_data n_lines:{n_lines}, lines:{lines}')

  return config, lines


# perform the translation of a source text thanks to a given model (a.k.a. LLM)
def get_translation(config, model, task, src_lines, dst_target_lines, verbose=False):

  if verbose:
    print('get_translation() src_lines:', src_lines)
    print('get_translation() dst_target_lines:', dst_target_lines)
  prompt = task.prompt

  print('get_translation() prompt:', prompt)

  text_src = ''
  
  # If any, add glossary lines in the beginning of the prompt
  if len(config.glossary_lines) > 0:
    print('get_translation() glossary_lines:', config.glossary_lines)

    text_src += '&&&\n'
    for line in config.glossary_lines:
        text_src += line+'\n'
    text_src += '&&&\n\n'

  line_separator = '\n'
  for line in src_lines:
    text_src += line + line_separator
  #text_src += list_to_str_blocks(line_separator, src_lines)
  text_dst_target = ''
  for line in dst_target_lines:
    text_dst_target += line + line_separator
  #list_to_str_blocks(line_separator, dst_target_lines)
  
  if verbose:
    print('get_translation() text_src:', text_src)
  
  error = False

  if 'gpt' in model:
    text_dst_predicted, total_tokens, price, error = openai.completion(config, model, prompt, text_src, text_dst_target)
  elif 'gemini' in model:
    text_dst_predicted, total_tokens, price, error = google.completion(config, model, prompt, text_src, text_dst_target)
  elif 'claude' in model:
    text_dst_predicted, total_tokens, price, error = anthropic.completion(config, model, prompt, text_src, text_dst_target)
  elif 'llama' in model:
    text_dst_predicted, total_tokens, price, error = llama.completion(config, model, prompt, text_src, text_dst_target)
  elif 'mistral' in model:
    text_dst_predicted, total_tokens, price, error = mistral.completion(config, model, prompt, text_src, text_dst_target)
  elif 'palm' in model:
    text_dst_predicted, total_tokens, price, error = palm.completion(config, model, prompt, text_src, text_dst_target)
  elif 'command-r' in model:
    text_dst_predicted, total_tokens, price, error = cohere.completion(config, model, prompt, text_src, text_dst_target)
  else:
      print(f'ERROR model {model} is not supported.')
      error = True
      return 'N/A', 0, 0, True

  if verbose:
    print('text_dst_predicted:', text_dst_predicted)

  text_dst_predicted = text_dst_predicted.rstrip()
  #dst_predic_lines = str_blocks_to_list(block_separator, text_dst_predicted)
  dst_predic_lines = text_dst_predicted.split('\n')
  error = False
  return dst_predic_lines, total_tokens, price, error


# launch the translation with a given model and estimate (i.e. score) the result
def translate(config, model, task, src_lines, verbose=False):
  
  if verbose:
    print('translate() src_lines:', src_lines)
  error = False

  # check that src and target text have same number of lines
  l1 = len(src_lines)

  dst_predic_lines, tokens, price, error = get_translation(config, model, task, src_lines, [])
  if error:
    error = True
    return None, error

  if verbose:
    print('n:', len(src_lines))
    print('src_lines:', src_lines)
    print('dst_predic_lines:', dst_predic_lines)

  l3 = len(dst_predic_lines)
  
  if verbose:
    print("++++++++++++++++++")
    print("src:"+'\\n'.join(src_lines)+" , n_lines:", l1)
    print("++++++++++++++++++")
    print("dst_p:"+'\\n'.join(dst_predic_lines)+" , n_lines:", l3)
    print("++++++++++++++++++")
  
  if len(dst_predic_lines) != len(src_lines):
    # if last line of scr_line contains '' and last line of dst_predic_lines contains '\n' 
    # then remove the last line of dst_predic_lines
    if src_lines[-1] == '' and dst_predic_lines[-1] == '\n':
      dst_predic_lines = dst_predic_lines[:-1]
    else:
      print(f'ERROR: with ${model}, number of lines in src and predicted text do not match: {len(src_lines)} vs {len(dst_predic_lines)}')
      print('dst_predic_lines:', dst_predic_lines)
      error = True
      return dst_predic_lines, tokens, price, error

  return dst_predic_lines, tokens, price, False

def eval(config, model, task, src_lines, dst_target_lines, dst_predic_lines, price, tokens, verbose=False):

  config, dst_target_lines = get_data(config, config.target_file)
  l1 = len(src_lines)
  l2 = len(dst_target_lines)
  l3 = len(dst_predic_lines)

  if verbose:
    print("++++++++++++++++++")
    print("src:"+''.join(src_lines)+" , n_lines:", l1)
    print("++++++++++++++++++")
    print("dst_t:"+''.join(dst_target_lines)+" , n_lines:", l2)
    print("++++++++++++++++++")
    print("dst_p:"+''.join(dst_predic_lines)+" , n_lines:", l3)
    print("++++++++++++++++++")

  if (l1 == l2) and (l2 == l3):
    is_translation_ok = True
  else:    
    is_translation_ok = False
    print(f'ERROR: number of lines in src, target and predicted text do not match: {l1} vs {l2} vs {l3}')
    return None, None, is_translation_ok

  df_details = pd.DataFrame()
  for line_src, line_dst_t, line_dst_p in zip(src_lines, dst_target_lines, dst_predic_lines):
    # if nothing in the source line (e.g. between 2 "paragraphs"), nothing to evaluate
    if line_src == '':
       continue
    # calculate number of words in the source sentence (just for info)
    br_words = len(line_src.split(' '))
    if line_dst_t == '':
      break
    score, scoring_tokens, scoring_price = scores.get_openai_score(config, line_dst_t, line_dst_p)

    sample_log = {
        'task': task.name,
        'model': model,
        'src': line_src,
        'target': line_dst_t,
        'prediction': line_dst_p,
        'score': score,
        'price': price + scoring_price,
        'n_tokens': tokens + scoring_tokens,
        'src_n_words': br_words
    }
    print('sample_log:')
    for key, value in sample_log.items():
        print(f"* {key}: {value}")
    df_sample = pd.DataFrame([sample_log])

    # Concatenate the new row DataFrame with the existing DataFrame
    df_details = pd.concat([df_details, df_sample], ignore_index=True)


    # only print the new line in the df (i.e. last line of the df)
    print('last row of df_model_results:', df_details.tail(1))

  try:
    score_mean = int(df_details['score'].mean()*100)/100
    score_std = int(df_details['score'].std()*100)/100
    price = int(df_details['price'].sum()*10000)/10000
    tokens = df_details['n_tokens'].sum()
    src_n_words_mean = int(df_details['src_n_words'].mean()*10)/10
    src_n_words_std = int(df_details['src_n_words'].std()*10)/10
  except:
    score_mean = pd.NA
    score_std = pd.NA
    score_mean = pd.NA
    score_std = pd.NA
    price = pd.NA
    tokens = pd.NA
    src_n_words_mean = pd.NA
    src_n_words_std = pd.NA

  result = {
    'task': task.name,
    'datetime': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
    'model': model,
    'score_mean': score_mean,
    'score_std': score_std,
    'price': price,
    'n_tokens': tokens,
    'src_n_words_mean': src_n_words_mean,
    'src_n_words_std': src_n_words_std
  }
  print(result)
  df_result = pd.DataFrame([result])
  error=False
  return df_details, df_result, error

# launch all the asks with each of supported models
def translate_and_eval(config, verbose=True):

  # prepare label for logs and results
  config, src_lines = get_data(config, config.source_file)

  if verbose:
    print('========================')
    print('text_src:', src_lines)
    print('------------------------')  

  label = config.source_file.replace('.txt', '')
  log_filename = label + '_' + config.log_file_postfix
  res_filename = label + '_' + config.res_file_postix
  
# create output file if not existing, otherwise open them
  if not os.path.exists(log_filename) or not os.path.exists(res_filename):
    df_full_results = pd.DataFrame()
    df_full_detailss = pd.DataFrame()
  else:
    if verbose:
      print('apppending results in file:', res_filename)
      print('appending logs in file:', log_filename)
    df_full_results = pd.read_csv(res_filename, sep='\t', encoding='utf-8')
    df_full_detailss = pd.read_csv(log_filename, sep='\t', encoding='utf-8')

  # test all the translation models listed is the config
  for model in config.models:          
    print(f'  * starting model {model}:')
    for task in config.tasks:
      print(f'  * starting task {task.name}:')
      assert(len(task.name) == 5)
      assert(task.name[2] == '2')
      config.lang_src = task.name[0:2]
      config.lang_dst = task.name[3:5]
        
      n_retry_max = 3
      n_try = 0
      # do translate() while n_retry < n_retry_max because sometimes
      # a translation will return a different number of lines than the source
      # so it is worth retrying
      while n_try < n_retry_max:
        n_try += 1
        dst_predic_lines, tokens, price, error = translate(config, model, task, src_lines)
        if error:
          print(f'ERROR: retrying {n_try}/{n_retry_max}...')
        else:
          break
      
      # save results, even if there is an error
      save_results(config, model, task, dst_predic_lines)

      if config.eval:
        print('evaluating...')
        config, dst_target_lines = get_data(config, config.target_file)
        df_details, df_results, error = eval(config, model, task, src_lines, dst_target_lines, dst_predic_lines, price, tokens)
        
        df_full_detailss = pd.concat([df_full_detailss, df_details], ignore_index=True)
        df_full_results = pd.concat([df_full_results, df_results], ignore_index=True)
        
        # sort results before writing them to the file
        df_full_results = df_full_results.sort_values(['task', 'score_mean'], ascending = [True, False])
        
        # append global results in the tsv results file    
        df_full_results.to_csv(res_filename, index=False, sep='\t', na_rep='n/a', encoding='utf-8')
        
        # append logs in the tsv logs file
        df_full_detailss.to_csv(log_filename, index=False, sep='\t', na_rep='n/a', encoding='utf-8')
        
        print(df_full_results)
      else:
        print(f'task:{task.name}, model:{model}, translated_lines:', dst_predic_lines)
        print('Given that target_file has not been found, no evaluation is done.')
        print('')
  return

def get_output_file(config, model, task):
    
  # remove the last 4 characters of the dataset source file
  filename = config.source_file[:-len('.txt')]
  
  # extract the last 2 characters of the task name
  target_lang = task.name[-2:]

  # get the currrent date and time under the form YYYY-MM-DD-HH-MM-SS
  now = datetime.now()
  date_time = now.strftime("%Y%m%d@%H%M%S")
   
  # create the output file name
  output_file = filename + '_'+ target_lang + '_' + model + '_' + date_time + '.txt'
  
  return output_file


def main(args):
  config = input_file.load_config(args)

  # trick to avoid reinstalling litellm each time there is a new llm model
  usage.update_model_price_list()

  translate_and_eval(config)
  print('finished')

if __name__ == '__main__':
    main(sys.argv)
