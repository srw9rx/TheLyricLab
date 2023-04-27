import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import requests

client_credentials_manager = SpotifyClientCredentials(client_id='your_client_id', client_secret='your_client_secret')
sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

results = sp.search(q='genre:"hip hop" year:2010-2021', type='track', limit=50)
tracks = results['tracks']['items']

for track in tracks:
    try:
        # Get the track's lyrics using Genius API
        song_title = track['name']
        artist_name = track['artists'][0]['name']
        lyrics_url = requests.get(f'https://api.genius.com/search?q={song_title} {artist_name}', headers={'Authorization': 'Bearer YOUR_ACCESS_TOKEN'}).json()['response']['hits'][0]['result']['url']
        lyrics_page = requests.get(lyrics_url)
        lyrics_text = lyrics_page.text.split('<div class="lyrics">')[1].split('</div>')[0].replace('<br/>', ' ').replace('\n', ' ').strip()

        # Save the lyrics to a file or a database
        with open('lyrics.txt', 'a') as f:
            f.write(lyrics_text + '\n')
    except:
        pass