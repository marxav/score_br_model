# score_br_model

## Goal
* This project is a quick and dirty project aiming at evaluating some large language models (LLMs) able to translate Breton language into French language ('br2fr' task), and vice versa ('fr2br' task).
* To do so, it compares the semantic distance of a translation performed by an LLM with an expected translation (a.k.a. target translation).
* The semantic distance is based on the proximity of OpenAI embeddings.
* Currently, only two LLMs can be tested: the latest OpenAI *gpt-3.5-turbo* and *gpt-4-turbo*.

## Requirements
* Ubuntu OS
* An OPENAI_API_KEY (cf. https://platform.openai.com/api-keys)

## Installation
* git clone https://github.com/marxav/score_br_model.git
* cd score_br_model
* python3 -m venv env
* source env/bin/activate
* pip install openai transformers torch sentencepiece pandas ipykernel
* echo OPENAI_API_KEY=your-secret-key > .env

## Run
* cd score_br_model
* source env/bin/activate
* python eval.py samples.tsv 

## More info
* The input file is either a *.tsv file (e.g. [samples.tsv](samples.tsv)). The *.tsv must contain two columns named 'br' and 'fr' and the columns must be separated by a tab (i.e. '\t').  
* Alternatively, the input_file is a *_br.txt file containing only breton sentences (e.g. [tregor_2110_br.txt](tregor_2110_br.txt)). In this case, another file *_fr.txt file containing only french sentences (i.e. the corresponding target sentences) (e.g. [tregor_2110_fr.txt](tregor_2110_fr.txt)); note that this second file must contain exactly the same number of sentences than the first *_br.tsv file.
* Running the eval.py creates 2 files 
  * A log file containing all translations and scores;
  * A result file containing the summary of scores.  
    * For example: 
    * [samples_logs.tsv](samples_logs.tsv)
    * [samples_res.tsv](samples_res.tsv)
  * To better view these 2 output files, you can use a jupyter notebook.
    * For example: 
    * [samples_logs_and_results.ipynb](samples_logs_and_results.ipynb).
  
## Todo
* Enhance the scoring metric(s)
* Add more samples in samples.tsv
* A leaderboard of the tested LLMs

## Acknowledgments
* [tregor_2110_br.txt](tregor_2110_br.txt) is a sample of a text written by Gireg Konan in Le Tregor newspaper, n°2110, June 6th 2024.
