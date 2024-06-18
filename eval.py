import os
import sys
from datetime import datetime
import pandas as pd
import scores
import input_file
from llms import anthropic, cohere, google, llama, mistral, openai

# https://docs.mistral.ai/getting-started/models/
# 'mistral-large-2402' <- 'mistral-large-latest'
# 'open-mixtral-8x22b-2404' <- 'open-mixtral-8x22b'
# https://docs.cohere.com/docs/models
 #'command-r-plus',
 #'open-mistral-7b', 'mistral-large-latest', # not (yet?) supported: 'open-mixtral-8x7b', #'open-mixtral-8x22b', 
 #'llama3-8b-8192', 'llama3-70b-8192',
 #'claude-3-haiku-20240307', 'claude-3-sonnet-20240229', 'claude-3-opus-20240229',
 #'gemini-1.5-flash', 'gemini-1.0-pro-001', 'gemini-1.5-pro-001', 
 #'gpt-3.5-turbo-0125', 'gpt-4-0613', 'gpt-4-turbo-2024-04-09', 'gpt-4o-2024-05-13'
 #'command-r-plus',
 #'llama3-70b-8192',
 #'mistral-large-2402',
#'claude-3-opus-20240229', 
#'gpt-4o-2024-05-13', 
#'gemini-1.5-pro-001',
#'gpt-4-turbo-2024-04-09'

def string_to_hex(s):
    return ' '.join(format(ord(c), '02x') for c in s)

def str_blocks_to_list(block_separator, input_string, verbose=False):

    # Split the string into blocks using '***' as the delimiter
    # 1 block either contains 1 row from the dataset (if row is short), 
    # or a part of a row (if the row is long)
    blocks = input_string.split(block_separator)
    
    # Remove any empty strings that may result from leading or trailing delimiters
    blocks = [s for s in blocks if s]

    if verbose:
        for block in blocks:
            hex_output = string_to_hex(block)
            print('block:', block)
            # display the block as a sequence of hexadecimal values
            print('block:', hex_output)

    # return the list of blocks
    return blocks

def list_to_str_blocks(block_separator, blocks):

    # Join the list of blocks with '***' and add '***' at the beginning and end
    result_string = block_separator + block_separator.join(blocks) + block_separator
    return result_string

# get the source data to be translated, as well as the ideal target data
def get_data(config, verbose=False):
    
    dataset_file = config.dataset_file
    lang_src = config.lang_src
    lang_dst = config.lang_dst
    
    if not os.path.isfile(dataset_file):
        print(f'warning: did not found the tsv input file')
    src_lines = []
    dst_target_lines = []
    
    df = pd.read_csv(dataset_file, sep='\t', encoding = 'utf8')
    df = df.dropna()
    print(df)

    len_df = df.shape[0]
    for index, row in df.iterrows():
        # Example texts
        if verbose:
            print(row)
            print('txt_src:', row[lang_src])
            print('txt_dst:', row[lang_dst])

        if len(row[lang_src]) > 100:
            if verbose:
                print('splitting line because line is too long')
            src_sub_lines = row[lang_src].replace('...', '…').split('.')
            dst_sub_lines = row[lang_dst].replace('...', '…').split('.')
            i = 0
            len_l = len(src_sub_lines)
            for (src_sub_line, dst_sub_line) in zip(src_sub_lines, dst_sub_lines):
                i+=1
                if i == len_l and src_sub_line == '\n':
                    print('warning: skip this last sub-line, otherwise gpt4-turbo does not like it')
                    continue
                if len(src_sub_line) > 0:
                    src_lines.append(src_sub_line)
                    dst_target_lines.append(dst_sub_line)
                    if verbose:
                        print('1', src_sub_line, len(src_sub_line))
                        print('2', dst_sub_line, len(dst_sub_line))
        else:
            src_lines.append(row[lang_src])
            dst_target_lines.append(row[lang_dst])

    return config, src_lines, dst_target_lines


# perform the translation of a source text thanks to a given model (a.k.a. LLM)
def get_translation(config, task, model, src_lines, dst_target_lines, verbose=True):

  lang_src = config.lang_src
  lang_dst = config.lang_dst
  block_separator = task.separator
  
  prompt = task.prompt
  text_src = list_to_str_blocks(block_separator,src_lines)
  text_dst_target = list_to_str_blocks(block_separator, dst_target_lines)
  
  if verbose:
    print('text_src:', text_src)
  
  error = False

  if 'gpt' in model:
    text_dst_predicted, total_tokens, price, error = openai.process(config, model, prompt, text_src, text_dst_target)
  elif 'gemini' in model:
    text_dst_predicted, total_tokens, price, error = google.process(config, model, prompt, text_src, text_dst_target)
  elif 'claude' in model:
    text_dst_predicted, total_tokens, price, error = anthropic.process(config, model, prompt, text_src, text_dst_target)
  elif 'llama' in model:
    text_dst_predicted, total_tokens, price, error = llama.process(config, model, prompt, text_src, text_dst_target)
  elif 'mistral' in model:
    text_dst_predicted, total_tokens, price, error = mistral.process(config, model, prompt, text_src, text_dst_target)
  elif 'command-r' in model:
    text_dst_predicted, total_tokens, price, error = cohere.process(config, model, prompt, text_src, text_dst_target)
  else:
      print(f'ERROR model {model} is not supported.')
      error = True
      return 'N/A', 0, 0, True

  if verbose:
    print('text_dst_predicted:', text_dst_predicted)

  text_dst_predicted = text_dst_predicted.rstrip()
  dst_predic_lines = str_blocks_to_list(block_separator, text_dst_predicted)
  error = False
  return dst_predic_lines, total_tokens, price, error


# launch the translation with a given model and estimate (i.e. score) the result
def test_model(config, task, translation_model, src_lines, dst_target_lines, verbose=True):
  
  error = False

  # check that src and target text have same number of lines
  l1 = len(src_lines)
  l2 = len(dst_target_lines)
  if l1 != l2:
    print(f'!!!warning len(sentences_src):{l1} is different from len(sentences_dst_t):{l2}')
    for i in range(min(l1, l2)):
        print(f'sentences_src[{i}]: {src_lines[i]}')
        print(f'sentences_dst[{i}]: {dst_target_lines[i]}')
        print('')
    return None, error

  is_translation_ok = False
  n_max = 3
  n = 0

  # here, we need a loop, because sometimes the translation ends up with more sentences in
  # the translated text than in the source text; hence we allow n_max trials be
  while is_translation_ok == False and n <= n_max:
    n += 1

    # perform the translation
    dst_predic_lines, tokens, price, error = get_translation(config, task, translation_model, src_lines, dst_target_lines)
    if error:
        error = True
        return None, error

    if verbose:
        print('n:', n)
        print('src_lines:', src_lines)
        print('dst_target_lines:', dst_target_lines)
        print('dst_predic_lines:', dst_predic_lines)

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


        if l3 > l1: # happens for llama and cohere for fr2br
            print(f'!!!warn len(lines_src):{l1} is different from len(lines_dst_p):{l3}')
            # remove lines afterwards
            dst_predic_lines = dst_predic_lines[l1:]
            is_translation_ok = True
            l3 = l1
        elif l1 != l3:
            print(f'!!!warning len(lines_src):{l1} is different from len(lines_dst_p):{l3}')
            i_min = min(l1, l3)
            i_max = max(l1, l3)
            for i in range(min(l1, l3)):
                print(f'src_lines[{i}]: {src_lines[i]}')
                print(f'dst_target_lines[{i}]: {dst_target_lines[i]}')
                print('')
            for i in range(i_min, i_max):
                if i_min == l1:
                    print(f'text_src[{i}]: n/a')
                    print(f'dst_predic_lines[{i}]: {dst_predic_lines[i]}')
                else:
                    print(f'src_lines[{i}]: {src_lines[i]}')
                    print(f'dst_predic_lines[{i}]: n/a')
        if n == n_max:

            print("ERROR, too many unsuccessful trials, let's stop here")
            error = True
            return None, error
  
  
  df_results = pd.DataFrame()
  for line_src, line_dst_t, line_dst_p in zip(src_lines, dst_target_lines, dst_predic_lines):

    # calculate number of words in br sentence (just for info)
    br_words = len(line_src.split(' '))
    if line_dst_t == '':
      break
    score = scores.get_openai_score(line_dst_t, line_dst_p)

    sample_log = {
        'task': task.name,
        'model': translation_model,
        'src': line_src,
        'target': line_dst_t,
        'prediction': line_dst_p,
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
def test_models(config, verbose=False):

    # prepare label for logs and results
    if config.dataset_file.endswith('.tsv'):
        label = config.dataset_file.replace('.tsv', '')
    else:
        label = config.dataset_file
        print(f"warning: dataset_file {label} does not finish with .tsv")
    
    log_filename = label + '_' + config.log_file_postfix
    res_filename = label + '_' + config.res_file_postix


    # test models for each of the task listed in the config
    for task in config.tasks: 
        print(f'* starting task {task}:')
        if task.name == 'br2fr':
            config.lang_src = 'br'
            config.lang_dst = 'fr'
        elif task.name == 'fr2br':
            config.lang_src = 'fr'
            config.lang_dst = 'br'
        else:
            print(f'Error: task {task.name} not implemented!')
            exit(-1)
              
        config, src_lines, dst_target_lines = get_data(config)

        if verbose:
            print('========================')
            print('text_src:', src_lines)
            print('text_dst_target:', dst_target_lines)
            print('------------------------')  
                
        # test all the translation models listed is the config
        for translation_model in config.models:          

            # create output file if not existing, otherwise open them
            if not os.path.exists(log_filename) or not os.path.exists(res_filename):
                df_full_results = pd.DataFrame()
                df_full_detailss = pd.DataFrame()
            else:
                print(res_filename)
                print(log_filename)
                df_full_results = pd.read_csv(res_filename, sep='\t', encoding='utf-8')
                df_full_detailss = pd.read_csv(log_filename, sep='\t', encoding='utf-8')

            print(f'  * starting translation_model {translation_model}:')
            df_details, error = test_model(config, task, translation_model, src_lines, dst_target_lines)
            if not error:
                score_mean = int(df_details['score'].mean()*100)/100
                score_std = int(df_details['score'].std()*100)/100
                price = int(df_details['price'].sum()*100)/100
                tokens = df_details['n_tokens'].sum()
                src_n_words_mean = int(df_details['src_n_words'].mean()*10)/10
                src_n_words_std = int(df_details['src_n_words'].std()*10)/10
            else:
                score_mean = pd.NA
                score_std = pd.NA
                price = pd.NA
                tokens = pd.NA
                src_n_words_mean = pd.NA
                src_n_words_std = pd.NA
            
            result = {
                'task': task.name,
                'datetime': datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                'model': translation_model,
                'score_mean': score_mean,
                'score_std': score_std,
                'price': price,
                'n_tokens': tokens,
                'src_n_words_mean': src_n_words_mean,
                'src_n_words_std': src_n_words_std
            }
            print(result)
            df_result = pd.DataFrame([result])

            df_full_detailss = pd.concat([df_full_detailss, df_details], ignore_index=True)
            df_full_results = pd.concat([df_full_results, df_result], ignore_index=True)

            # sort results before writing them to the file
            df_full_results = df_full_results.sort_values(['task', 'score_mean'], ascending = [True, False])
        
            # append global results in the tsv results file    
            df_full_results.to_csv(res_filename, index=False, sep='\t', na_rep='n/a', encoding='utf-8')

            # write logs in a second file which name contains the date
            #dated_res_filename = result['datetime'] + '_' + label + '_' + config.res_file_postix
            #df_full_results.to_csv(dated_res_filename, index=False, sep='\t', encoding='utf-8')

            # append logs in the tsv logs file
            df_full_detailss.to_csv(log_filename, index=False, sep='\t', na_rep='n/a', encoding='utf-8')
            # write logs in a second file which name contains the date
            #dated_log_filename = result['datetime'] + '_' + label + '_' + config.log_file_postfix
            #df_details.to_csv(dated_log_filename, index=False, sep='\t', encoding='utf-8')
            
    return df_full_results
    
def main(args):
    config = input_file.load_config(args)
    test_results = test_models(config)
    print('test_results:', test_results)

if __name__ == '__main__':
    main(sys.argv)