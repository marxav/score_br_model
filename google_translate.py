from google.cloud import translate_v2 as translate
import os

# Set up Google Cloud credentials
username= os.getlogin()
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/'+username+'/.config/gcloud/application_default_credentials.json'

# Initialize the Google Translate client
translate_client = translate.Client()

def translate_text(config, text_src, verbose=False):

    task = config.tasks[0]
    source_language = task.name[0:2]
    target_language = task.name[3:5]

    """
    Translate text using Google Translate API.
    """
    results = ''
    for sentence in text_src.split('\n'):
        result = translate_client.translate(
            sentence,
            source_language=source_language,
            target_language=target_language
        )
        results += result['translatedText'] + '\n'
    if verbose:
        print('results:', results)
        
    return results
