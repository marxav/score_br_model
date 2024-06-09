# score_br_model

## Goal
* This project is a quick and dirty project aiming at evaluating an LLM able to translate Breton language into French language.
* To do so, it compares the semantic distance of a breton sentence translated in French with a target translation.
* The semantic distance is based on cosine similarity between the Camenbert vector emmbedding of the target french sentence and the vector emmbedding of the french sentence translated by the model.
* Currently, the only two models that can be tested are OpenAI *gpt-3.5-turbo* and *gpt-4-turbo*

## Requirements
* Ubuntu OS
* An *OPENAI_API_KEY* (cf. https://platform.openai.com/api-keys)

## Installation
* git clone https://github.com/marxav/score_br_model.git
* cd score_br_model
* python3 -m venv env
* source env/bin/activate
* python -m pip install -r requirements.txt
* echo OPENAI_API_KEY=your-secret-key > .env

## Run
* cd score_br_model
* source env/bin/activate
* python eval.py samples.csv 

## More info
* The input file is either a *.csv file containing two columns named 'Brezhoneg' and 'Français' separated by a tab (i.e. '\t'). [samples.csv](samples.csv) is an example of such a file.
* Alternatively, the input_file is a *_br..txt file containing only breton sentences. In this case, another file *_fr.txt file containing only french target sentences and exactly the same number of sentences than the foo_br.csv file. [tregor_2110_br.txt](tregor_2110_br.txt) and [tregor_2110_fr.txt](tregor_2110_fr.txt) represent an example of such files.
* The eval creates 2 files 
  * 1 log file containing all translations and scores;
  * 1 result file containing the summary of scores.  
    * For example: 
    * [2024-06-09_22:49:46_samples_logs.csv](2024-06-09_22:49:46_samples_logs.csv).
    * [2024-06-09_22:49:46_samples_res.csv](2024-06-09_22:49:46_samples_res.csv).
  * To better view these 2 files, you can use a jupyter notebook.
    * For example: 
    * [samples_csv_logs_and_results.ipynb](samples_csv_logs_and_results.ipynb).
  
## Todo
* Add evaluation scores for fr->br
* Enhance the scoring metric(s)
* Add more samples in samples.csv
* A leaderboard of the tested LLMs

## Acknowledgments
* The text [tregor_2110_br.txt](tregor_2110_br.txt) was written by Gireg Konan and comes from Le Tregor, n°2110, June 6th 2024.
