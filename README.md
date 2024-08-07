# score_br_model

## Goal
* This project is a quick and dirty tool to evaluate some large language models (LLMs) in their ability to carry out tasks via interaction in Breton language. 
* So far, only 2 tasks are implemented:
  * *br2fr* (Breton to French translation)
  * *fr2br* (French to Breton translation)
* The evaluation produces a proximity *score* comparing the semantic distance of a text produced by an LLM with an expected text pre-written by a human evaluator.
* The semantic distance is based on the proximity of OpenAI embeddings. 
* Currently, the models from the following providers can be tested: 
  * [OpenAI](https://platform.openai.com/docs/models/): e.g. *gpt-3.5-turbo*, *gpt-4-turbo*, *gpt-4o*, *gpt-4o-mini-2024-07-18*
  * [Google](https://ai.google.dev/gemini-api/docs/models/gemini) e.g. *gemini-1.0-pro*, *gemini-1.5-flash*, *gemini-1.5-pro*, *palm-2-chat-bison-32k*
  * [Anthropic](https://docs.anthropic.com/en/docs/models-overview): e.g. *claude-3-5-sonnet-20240620*, *claude-3-haiku-20240307*, *claude-3-sonnet-20240229*, *claude-3-opus-20240229*
  * [Meta](https://console.groq.com/docs/models): e.g. *llama-3.1-70b-versatile*, *llama-3.1-8b-instant*, *llama3-8b-8192*, *llama3-70b-8192*
  * [Mistral](https://docs.mistral.ai/getting-started/models/) *open-mistral-7b*, *mistral-large-latest*
  * [Cohere](https://docs.cohere.com/docs/models): e.g. *command-r-plus*
  

## Requirements
* Ubuntu OS
* An OPENAI_API_KEY (cf. [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys))
* A GOOGLE_API_KEY (cf. [(https://ai.google.dev/gemini-api/docs/api-key](https://ai.google.dev/gemini-api/docs/api-key))
* An ANTHROPIC_API_KEY (cf. [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys))
* A GROQ_API_KEY (cf. [https://console.groq.com/keys](https://console.groq.com/keys)) for Llama models
* A MISTRAL_API_KEY (cf. [https://console.mistral.ai/api-keys/](https://console.mistral.ai/api-keys/))
* A COHERE_API_KEY (cf. [https://dashboard.cohere.com/api-keys](https://dashboard.cohere.com/api-keys))
* A OPENROUTER_API_KEY (cf. [https://openrouter.ai/keys](https://openrouter.ai/keys)) for Google Palm models
* Only the OPENAI_API_KEY is mandatory given it is also needed for calculating the evaluation scores.
* A mandatory source file of your choice (e.g. [samples_br.txt](samples_br.txt))
* An optional target file of your choice (e.g. [samples_fr.txt](samples_fr.txt)). If not provided, evaluation will not be performed.
* a dedicated configuration file (e.g [samples_br.yaml](samples_br.yaml))

## Installation
* git clone https://github.com/marxav/score_br_model.git
* cd score_br_model
* python3 -m venv env
* source env/bin/activate
* pip install openai pandas ipykernel tabulate llmlite google-generativeai anthropic groq mistralai cohere
* echo OPENAI_API_KEY=your-secret-key-1 >> .env
* echo GOOGLE_API_KEY=your-secret-key-2 >> .env
* echo ANTHROPIC_API_KEY=your-secret-key-3 >> .env
* echo GROQ_API_KEY=your-secret-key-4 >> .env
* echo MISTRAL_API_KEY=your-secret-key-5 >> .env
* echo COHERE_API_KEY=your-secret-key-6 >> .env
* echo OPENROUTER_API_KEY=your-secret-key-7 >> .env

## Run
* cd score_br_model
* source env/bin/activate
* python translate_and_eval.py samples.yaml 

## Results
* The result file will contain something like :

| task   | model                  | score       |
|:-------|:-----------------------|:------------|
| br2fr  | claude-3-opus-20240229 | 0.88 ± 0.18 |
| br2fr  | gemini-1.5-pro-001     | 0.84 ± 0.17 |
| br2fr  | gpt-4o-2024-05-13      | 0.81 ± 0.17 |
| br2fr  | gpt-4-turbo-2024-04-09 | 0.8 ± 0.21  |
| br2fr  | gemini-1.5-flash       | 0.78 ± 0.19 |
| br2fr  | llama3-70b-8192        | 0.74 ± 0.18 |
| br2fr  | gpt-3.5-turbo-0125     | 0.66 ± 0.16 |
| br2fr  | command-r-plus         | 0.58 ± 0.12 |
| br2fr  | mistral-large-2402     | 0.57 ± 0.15 |

| task   | model                  | score       |
|:-------|:-----------------------|:------------|
| fr2br  | claude-3-opus-20240229 | 0.69 ± 0.14 |
| fr2br  | gpt-4-turbo-2024-04-09 | 0.68 ± 0.17 |
| fr2br  | gpt-4o-2024-05-13      | 0.65 ± 0.17 |
| fr2br  | gemini-1.5-flash       | 0.64 ± 0.18 |
| fr2br  | gemini-1.5-pro-001     | 0.61 ± 0.14 |
| fr2br  | mistral-large-2402     | 0.58 ± 0.14 |
| fr2br  | gpt-3.5-turbo-0125     | 0.57 ± 0.16 |
| fr2br  | llama3-70b-8192        | 0.25 ± 0.16 |
| fr2br  | command-r-plus         | 0.23 ± 0.18 |

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
* Some model can refuse to translate some sentences that they consider as :
  * *HARM CATEGORY_SEXUALLY_EXPLICIT*,
  * *HARM_CATEGORY_HATE_SPEECH*,
  * *HARM_CATEGORY_HARASSMENT*,
  * *HARM_CATEGORY_DANGEROUS_CONTENT*.
* Currently, we use a file with lines like "br:port nawak" and "fr:trop chouette"
  * For the br2fr task "br:port nawak" is sent to the model; the answer is then compared with "fr:trop chouette", which is used as true value.
  * For the fr2br task "fr:trop chouette" is sent to the model; the answer is then compared with "br:port nawak", which is used as true value.
  * A model saving history could then learn that "br:port nawak" has to be translated by fr:trop chouette", which would bias the evaluation.
  * TODO: try to check if this happens... or avoid using the same input-file for the two different tasks.

## Other information
* Instead of using this tool, you can manually use [LMSYS](https://chat.lmsys.org) (in the "Arena side-by-side" tab) to compare the results of 2 models
  * In the parameters, set *temperature=0.0* and *top_p=0.95*
  * For the *br2fr* task, input a [prompt](https://arxiv.org/pdf/2406.06608) like:
    * *Translate the following Breton text to French. Immediatly write the translated text, nothing more. Do not add any personal comment beyond translation, just translate. The translated text must contain the same number of sentences and same number of '.' characters as in the input text.\n\nC'hoant am eus da ganañ. Ar wirionez zo gantañ. Ar c'hi zo bras awalc'h. Na vezit ket e gortoz e rofen ar respontoù deoc’h. Echu eo.*
  * For the *fr2br* task, input a [prompt](https://arxiv.org/pdf/2406.06608) like:
    * *Translate the following French text to Breton. Immediatly write the translated text, nothing more. Do not add any personal comment beyond translation, just translate. The translated text must contain the same number of sentences and same number of '.' characters as in the input text.\n\nJ'ai envie de chanter. Il a raison. Le chien est assez grand. Ne vous attendez pas à ce que je vous donne les réponses. C'est fini.*

## Acknowledgments
* [tregor_2110_br.txt](/examples/peurunvan/tregor_2110_br.txt) is a sample of a text written by Gireg Konan in Le Tregor newspaper, n°2110, June 6th 2024.
