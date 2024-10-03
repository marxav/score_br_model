# score_br_model

## Goal
* This project is a quick and dirty tool to evaluate some large language models (LLMs) in their ability to carry out tasks via interaction in Breton language. 
* So far, only 2 tasks are implemented:
  * *br2fr* (Breton to French translation)
  * *fr2br* (French to Breton translation)
* The evaluation produces a proximity *score* comparing the semantic distance of a text produced by an LLM with an expected text pre-written by a human evaluator.
* The semantic distance is based on the proximity of OpenAI embeddings. 
* Currently, the models from the following providers can be tested: 
  * [Anthropic](https://docs.anthropic.com/en/docs/models-overview): e.g. *claude-3-5-sonnet-20240620*, *claude-3-haiku-20240307*, *claude-3-sonnet-20240229*, *claude-3-opus-20240229*
  * [Cohere](https://docs.cohere.com/docs/models): e.g. *command-r-plus* 
  * [Google](https://ai.google.dev/gemini-api/docs/models/gemini) e.g. *gemini-1.0-pro*, *gemini-1.5-flash*, *gemini-1.5-pro*, *palm-2-chat-bison-32k*
  * [Google Translate](https://cloud.google.com/translate/docs/advanced/translating-text-v3) e.g. *google-translate* (it is not strictly an LLM as other models (no prompt allowed), but can be tested nevertheless)
  * [Meta](https://console.groq.com/docs/models): e.g. *llama-3.1-70b-versatile*, *llama-3.1-8b-instant*, *llama3-8b-8192*, *llama3-70b-8192*
  * [Mistral](https://docs.mistral.ai/getting-started/models/) *open-mistral-7b*, *mistral-large-latest*
  * [OpenAI](https://platform.openai.com/docs/models/): e.g. *gpt-3.5-turbo*, *gpt-4-turbo*, *gpt-4o*, *gpt-4o-mini-2024-07-18*
  
## Requirements
* Ubuntu OS
* An OPENAI_API_KEY (cf. [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys))
* A GOOGLE_API_KEY (cf. [(https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key))
* An ANTHROPIC_API_KEY (cf. [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys))
* A GROQ_API_KEY (cf. [https://console.groq.com/keys](https://console.groq.com/keys)) for Llama models
* A MISTRAL_API_KEY (cf. [https://console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/))
* A COHERE_API_KEY (cf. [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys))
* A OPENROUTER_API_KEY (cf. [https://openrouter.ai/keys](https://openrouter.ai/keys)) for Google Palm models
* A GOOGLE_TRANSLATION_PROJECT_ID (cf. https://console.cloud.google.com/, assign a project to Cloud Translation API)
* Only the OPENAI_API_KEY is mandatory given it is also needed for calculating the evaluation scores.
* A mandatory source file of your choice (e.g. [samples_br.txt](samples_br.txt))
* An optional target file of your choice (e.g. [samples_fr.txt](samples_fr.txt)). If not provided, evaluation will not be performed.
* a dedicated configuration file (e.g [samples_br.yaml](samples_br.yaml))

## Installation
* git clone https://github.com/marxav/score_br_model.git
* cd score_br_model
* python3 -m venv env
* source env/bin/activate
* pip install openai pandas ipykernel tabulate google-generativeai anthropic groq mistralai cohere jupyter google-cloud-translate
* echo OPENAI_API_KEY=your-secret-key-1 >> .env
* echo GOOGLE_API_KEY=your-secret-key-2 >> .env
* echo ANTHROPIC_API_KEY=your-secret-key-3 >> .env
* echo GROQ_API_KEY=your-secret-key-4 >> .env
* echo MISTRAL_API_KEY=your-secret-key-5 >> .env
* echo COHERE_API_KEY=your-secret-key-6 >> .env
* echo OPENROUTER_API_KEY=your-secret-key-7 >> .env
* echo GOOGLE_TRANSLATION_PROJECT_ID=your-secret-gt-project-id >> .env
* to test google-translate 
 * log to https://console.cloud.google.com/ 
   * Enable *Cloud Translation API*
 * go to https://cloud.google.com/sdk/docs/install and install *gcloud CLI* on your machine
 * and then 
   * gcloud init
   * gcloud auth application-default login

## Run
* cd score_br_model
* source env/bin/activate
* python translate_and_eval.py samples.yaml 

## Results
* The result file will contain something like :

| task   | model                                         | score       |   s_rank |       price |   p_rank |
|:-------|:----------------------------------------------|:------------|---------:|------------:|---------:|
| br2fr  | openai/gpt-4-turbo-preview                    | 0.97 ± 0.05 |        1 | 0.00378     |      136 |
| br2fr  | openai/gpt-4-vision-preview                   | 0.97 ± 0.05 |        2 | 0.00378     |      137 |
| br2fr  | openai/gpt-4-1106-preview                     | 0.97 ± 0.05 |        3 | 0.00384     |      138 |
| br2fr  | openai/gpt-4o:extended                        | 0.95 ± 0.07 |        6 | 0.001968    |      122 |
| br2fr  | anthropic/claude-3-haiku                      | 0.95 ± 0.07 |        4 | 0.00017025  |       79 |
| br2fr  | anthropic/claude-3-haiku:beta                 | 0.95 ± 0.07 |        5 | 0.00017025  |       80 |
| br2fr  | anthropic/claude-3-opus                       | 0.95 ± 0.06 |        7 | 0.01029     |      140 |
| br2fr  | anthropic/claude-3-opus:beta                  | 0.95 ± 0.07 |        8 | 0.01029     |      141 |
| br2fr  | google/palm-2-chat-bison-32k                  | 0.94 ± 0.1  |        9 | 0.000304    |       93 |
| br2fr  | openai/gpt-4o-2024-05-13                      | 0.93 ± 0.08 |       10 | 0.00161     |      120 |
| br2fr  | openai/gpt-4-turbo                            | 0.93 ± 0.08 |       11 | 0.00366     |      135 |
| br2fr  | openai/chatgpt-4o-latest                      | 0.92 ± 0.07 |       13 | 0.0017      |      121 |
| br2fr  | openai/gpt-4o-2024-08-06                      | 0.92 ± 0.1  |       12 | 0.00094     |      114 |
| br2fr  | anthropic/claude-3.5-sonnet                   | 0.92 ± 0.17 |       14 | 0.002073    |      124 |
| br2fr  | anthropic/claude-3.5-sonnet:beta              | 0.92 ± 0.17 |       15 | 0.002073    |      125 |
| br2fr  | openai/o1-preview-2024-09-12                  | 0.91 ± 0.1  |       16 | 0.117255    |      144 |
| br2fr  | google/gemini-pro-1.5-exp                     | 0.9 ± 0.14  |       17 | 0           |        1 |
| br2fr  | google/gemini-flash-1.5-exp                   | 0.89 ± 0.1  |       18 | 0           |        2 |
| br2fr  | nousresearch/hermes-3-llama-3.1-405b:free     | 0.89 ± 0.15 |       19 | 0           |        3 |
| br2fr  | perplexity/llama-3.1-sonar-huge-128k-online   | 0.89 ± 0.14 |       20 | 0.001095    |      116 |
| br2fr  | anthropic/claude-3-sonnet                     | 0.89 ± 0.16 |       21 | 0.002073    |      126 |
| br2fr  | anthropic/claude-3-sonnet:beta                | 0.89 ± 0.16 |       22 | 0.002073    |      127 |
| br2fr  | google/gemini-pro-vision                      | 0.88 ± 0.12 |       23 | 0.0001775   |       83 |
...

| task   | model                      | score       |
|:-------|:---------------------------|:------------|
| fr2br  | gpt-4-turbo-2024-04-09     | 0.73 ± 0.18 |
| fr2br  | claude-3-5-sonnet-20240620 | 0.72 ± 0.16 |
| fr2br  | gpt-4o-2024-08-06          | 0.71 ± 0.15 |
| fr2br  | llama-3.1-70b-versatile    | 0.7 ± 0.13  |
| fr2br  | palm-2-chat-bison-32k      | 0.7 ± 0.19  |
| fr2br  | google-translate           | 0.69 ± 0.14 |
| fr2br  | claude-3-opus-20240229     | 0.68 ± 0.1  |
| fr2br  | gemini-1.5-flash           | 0.68 ± 0.16 |
| fr2br  | gemini-1.5-pro             | 0.67 ± 0.11 |
| fr2br  | gpt-4o-2024-05-13          | 0.65 ± 0.19 |
| fr2br  | mistral-large-2407         | 0.65 ± 0.14 |
| fr2br  | llama3-70b-8192            | 0.64 ± 0.18 |
| fr2br  | gemini-1.0-pro             | 0.63 ± 0.15 |
| fr2br  | gpt-3.5-turbo-0125         | 0.58 ± 0.06 |
| fr2br  | mistral-large-2402         | 0.58 ± 0.17 |
| fr2br  | gpt-4o-mini-2024-07-18     | 0.57 ± 0.17 |
| fr2br  | llama-3.1-8b-instant       | 0.56 ± 0.15 |
| fr2br  | open-mistral-nemo-2407     | 0.51 ± 0.16 |
| fr2br  | command-r-plus             | 0.5 ± 0.12  |

## More info
* The source text to be translated must be in a *.txt file (e.g. [samples_br.txt](samples_br.txt)). 
* In order to evaluate the translation, another file must contain the target translation  (e.g. [samples_fr.txt](samples_fr.txt)), to which the translation will be compared to carry out the evaluation.
* Running the translate_and_eval.py creates 2 files 
  * A log file containing all translations and scores;
    * For example: [samples_br_logs.tsv](samples_br_logs.tsv)
  * A result file containing the summary of scores.  
    * For example: [samples_br_res.tsv](samples_br_res.tsv)
  
## Todo
* Enhance the scoring metric(s)
* Add more samples in samples.tsv
* Add a leaderboard of the tested LLMs and theirs scores at different tasks
  * Either like an [LMSYS](https://chat.lmsys.org/?leaderboard) leaderboard
  * Or with via a product like [https://scale.com/leaderboard](https://scale.com/leaderboard)

## Warning
* Some models can refuse to translate some sentences that they consider as :
  * *HARM CATEGORY_SEXUALLY_EXPLICIT*,
  * *HARM_CATEGORY_HATE_SPEECH*,
  * *HARM_CATEGORY_HARASSMENT*,
  * *HARM_CATEGORY_DANGEROUS_CONTENT*.

## Other information
* Add the "virtual" '**openrouter/all**' model in the yaml file if you want to test all LLMs available at [OpenRouter.ai](https://openrouter.ai/api/v1/models)
* Instead of using this tool, you can manually use [LMSYS](https://chat.lmsys.org) (in the "Arena side-by-side" tab) to compare the results of 2 models
  * In the parameters, set *temperature=0.0* and *top_p=0.95*
  * For the *br2fr* task, input a [prompt](https://arxiv.org/pdf/2406.06608) like:
    * *Translate the following Breton text to French. Immediatly write the translated text, nothing more. Do not add any personal comment beyond translation, just translate. The translated text must contain the same number of sentences and same number of '.' characters as in the input text.\n\nC'hoant am eus da ganañ. Ar wirionez zo gantañ. Ar c'hi zo bras awalc'h. Na vezit ket e gortoz e rofen ar respontoù deoc’h. Echu eo.*
  * For the *fr2br* task, input a [prompt](https://arxiv.org/pdf/2406.06608) like:
    * *Translate the following French text to Breton. Immediatly write the translated text, nothing more. Do not add any personal comment beyond translation, just translate. The translated text must contain the same number of sentences and same number of '.' characters as in the input text.\n\nJ'ai envie de chanter. Il a raison. Le chien est assez grand. Ne vous attendez pas à ce que je vous donne les réponses. C'est fini.*

## Acknowledgments
* [tregor_2110_br.txt](/examples/peurunvan/tregor_2110_br.txt) is a sample of a text written by Gireg Konan (Le Tregor newspaper, n°2110, June 6th 2024).
* [assimil_76_br.txt](/examples/etrerannyezhel/assimil_76_br.txt) is a sample of a text written by Fañch Morvannou (Le Breton sans peine Assimil book, 1990).
* [maria_prat_br.txt](/examples/tregerieg/maria_prat_br.txt) is a sample of a text written by Maria Prat (Eun toullad kontadennou, Skol Uhel ar Vro, 1988).
* [pipi_gonto_br.txt](/examples/tregerieg/pipi_gonto_br.txt) is a sample of a text written by Dir na dor (Pipi Gonto, E. Kemper, 1926).
