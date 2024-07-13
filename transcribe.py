from __future__ import print_function
import sys
import pickle
import os.path
import base64
import requests
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pandas as pd
from openai import OpenAI
    
open_api_key = next((line.split('=')[1].strip() for line in open('.env') if line.startswith('OPENAI_API_KEY')), None)

def transcribe(filename, local_filename):
    """
    Transcribes a Breton text from an image using OpenAI's GPT-4o model.

    Args:
        filename (str): The name of the image file.
        local_filename (str): The local path to the image file.

    Returns:
        None
    """
    directory = 'transcription-results'
    # if directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print('local_filename:', local_filename)
    # Function to encode the image
    def encode_image(image_path):
        print('image_path:', image_path)
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    # Getting the base64 string
    base64_image = encode_image(local_filename)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {open_api_key}"
    }

    payload = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
        {
            "role": "user",
            "content": [
            {
                "type": "text",
                "text": "Transcribe the Breton text from the image. \
                    Do not add anything besides translation. \
                    Take care preserving the 'ñ' and 'ê' characters."
            },
            {
                "type": "image_url",
                "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
            ]
        }
        ],
        "max_tokens": 3000,
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())
    data = response.json()
    text = data["choices"][0]["message"]["content"]
    
    filename = filename.replace('.jpg', '.txt')
    filename = os.path.join(directory, filename)
    with open(filename, 'w') as f:
        f.write(text)
        print('transcription saved to:', filename)
    


def download(filename, base_url):
    """
    Downloads an image from a Google Photos URL and saves it locally.

    Args:
        filename (str): The name of the image file.
        base_url (str): The base URL of the image.

    Returns:
        str: The local path to the downloaded image file.
    """
    assert(base_url.startswith('https://lh3.googleusercontent.com'))
    assert(filename.endswith('.jpg'))
    
    directory = 'transcription-photos'
    # if directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    #base_url = "https://lh3.googleusercontent.com/lr/AAJ1LKdMyLPoi8fRmthcm5MDwOglxrwAgzsxAu0oNLiZKHU4XfNJY40BtNb7YhGIQ0pabUOBsoPHgDIra94COFtE3vorjvbwdiQlgNXUdpZMO9Qn_gdOGggsFu8rxFxhXYqXdXLg3DWBjmRNLuXt8qVMLveG2Uz79tODAlWJK4P6Pk-dtPxNaU5fu20Q9h0uR_E-sXwds02MPU8KGkXkd01BHM4KS4DBzNjh5skymXj0Dv1f2YFkhcFmC9Cx6XriwuuSHblpxsWoy38PwdewY2NBEVJH5ASiFCNlJAoqz3pgJs1dZP_Y9gHueTUqrxSvqirSTnyeFxfJMpgyBnF6HkoDEiFnsaZVjr_xljtlZqlzUZo1h5gxVaH4HLCTEWHEMBCCAiRPPQoW3WCSnDk5jd-F1dtRXDeE9ulUPfSXN1iOfak30HFeiq9XxUqD87Uw7Bl3oxw562g2xCOoVJwh1Fyq-3B1INY7sysJ4KVQCgg45AeDfPPQyVpkUDyskh1zsfFYj79n56Cc_4ofGI9Xrajm67Xqxe3RC2WOziSUMcK-MHzF3JZRLo3eCFWpR_PIjbzki-Y5nDuFXIPB2XUYnFl2OI-VTVxbD7Ukl8L0H6VVZabCNMoHzoQHoCgbpYwPgwYjrlm2TljGWzNWs6KkYG5Ag_D8krt-7Rjal4q9YAKP7a-XzcMGl7KXmhXFvM_Mx3ZOH8A9bJdQviCSEaUNBz98SaT9tQ7GWYzuRYVlGKSzqFsAsR13PGwqeniieccKdHtgxYDCsfZ-LTRRzfqGwmvBPp_8b95vCtKWTyqjB4I6BjePfUxGo06lgb2quQbIX_gi9dWKfWSBFmDrFPcfS0zwtH3sI0rtFZ3bDaqJjKBFz-en_0dipa3UIGrscaBwpegfvZE-4fKdyUNuWubfCSlx7aExCahCQCLw0aFXPP6gGmA7Prq37W0wXjEsvFKkjGuvrM--ikOIrC3FhdGZEVhb67Lj3Tek8WtN51mO949wh3ieyvqyafdmxD2NTv0iFc-yMr_EaNlND-rCMHVvJvhMICbI"
    #product_url = 'https://photos.google.com/lr/album/AAFP53Rjo1NtXCr9ZywG4Ec4c8xJkU13TarK_AhuZE5WxUAGqjCuKzfPjVXQDSssApPXQy1dJtI_/photo/AAFP53QrrHPiFNAzJD1i6ekiW6ledzpJQ5mhR87MeK4yxjdwdyJc9-KHZOg2yZUotM1_2eggRmJCe1qBYOvzInOQ0yRBIuRNVw'
    
    # add a postfix parameter to base_url to indicate a better resolution
    # cf. https://developers.google.com/photos/library/guides/access-media-items?hl=en
    base_url = base_url + '=d' 
    
    # download the image from its url
    image = requests.get(base_url)
    # save the image to a file
    local_filename = os.path.join(directory, filename)
    with open(local_filename, 'wb') as f:
        f.write(image.content)
        
    return local_filename
        

def main(argv):
    """
    Main function that orchestrates the transcription process.
    It fetches a given album within the list of albums from Google Photos 
    and transcribes the Breton text from the images in this album.

    Args:
        str: The partial name of an album in Google Photos
    Returns:
        None
    """
    album_name = argv[1]
    
    directory = 'transcription-results'
    # if directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)
        
    # contains the list of already processed photos
    processed_photos_file = os.path.join(directory, album_name+'.csv')
    
    # check if the file exists
    if os.path.exists(processed_photos_file):
        print('found processed_photos_file:', processed_photos_file)
        df_processed_photos = pd.read_csv(processed_photos_file)
    else:
        # create an empty DataFrame
        print('not found processed_photos_file:', processed_photos_file)
        df_processed_photos = pd.DataFrame()
        df_processed_photos.to_csv(processed_photos_file, index=False)
    
    credentialsFile = 'credentials.json'  # Please set the filename of credentials.json
    pickleFile = 'token.pickle'  # Please set the filename of pickle file.

    SCOPES = ['https://www.googleapis.com/auth/photoslibrary']
    creds = None
    if os.path.exists(pickleFile):
        with open(pickleFile, 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentialsFile, SCOPES)
            creds = flow.run_local_server()
        with open(pickleFile, 'wb') as token:
            pickle.dump(creds, token)

    service = build('photoslibrary', 'v1', credentials=creds, static_discovery=False)

    # Call the Photo v1 API
    results = service.albums().list(
        pageSize=10, fields="nextPageToken,albums(id,title)").execute()
    items = results.get('albums', [])
    if not items:
        print('No albums found.')
    else:
        print('Albums:')
        for item in items:
            if album_name in item['title']:
                print('{0} ({1})'.format(item['title'].encode('utf8'), item['id']))
                print('item.key:', item.keys)
                
                # print all photo names contained in  the album, page size max 200
                nextPageToken = None
                stop = False
                n_photos = 0
                df = pd.DataFrame()
                # while nextPageToken is not None, keep fetching the next page
                while not stop and n_photos < 1000: # HARDCODED LIMIT to 1000 photos !
                    results = service.mediaItems().search(body={"albumId": item['id'], "pageSize": 50, "pageToken": nextPageToken}).execute()
                    n_photos = len(results["mediaItems"])
                    nextPageToken = results.get('nextPageToken') 
                    if nextPageToken is None:
                        stop = True
                    print('n_photos:', n_photos)  
                
                    for r in results['mediaItems']:
                        photo = {
                            'filename': r['filename'],
                            'creationTime':  r['mediaMetadata']['creationTime'],
                            'baseUrl': r['baseUrl']
                        }
                        print("photo['filename']:", photo['filename'])
                        if photo['filename'] not in df_processed_photos.values:
                            print('r:', r)
                            print('filename:', photo['filename'], 'not in df_processed_photos')
                            print('len(baseUrl):', len(r['baseUrl']))
                            local_filename = download(photo['filename'], photo['baseUrl'])
                            transcribe(photo['filename'], local_filename)
                            dfi = pd.DataFrame([{'filename': photo['filename']}], index=[0])
                            df_processed_photos = pd.concat([df_processed_photos, dfi], ignore_index=True)
                            df_processed_photos.to_csv(processed_photos_file, index=False)
                
if __name__ == '__main__':
    main(argv=sys.argv)