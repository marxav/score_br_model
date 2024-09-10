from google.cloud import translate_v3 

project_id = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('GOOGLE_TRANSLATION_PROJECT_ID')), None)

client = translate_v3.TranslationServiceClient()
location = "global"
parent = f"projects/{project_id}/locations/{location}"


# Initialize Translation client with translate API v3
def translate_text(config, text_src, verbose=False):
    task = config.tasks[0]
    source_language = task.name[0:2]
    target_language = task.name[3:5]

    # More info on Google Translate v3 API at
    # https://cloud.google.com/translate/docs/advanced/translating-text-v3
    results = ''
    for line in text_src.rstrip().split('\n'):
        if verbose:
            print("line:", line)
        response = client.translate_text(
            request={
                "parent": parent,
                "contents": [line],
                "mime_type": "text/plain",  
                "source_language_code": source_language,
                "target_language_code": target_language,
            }
        )
        
        results += response.translations[0].translated_text + '\n'
    if verbose:
        print('results:', results)
        
    return results
