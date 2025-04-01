import requests
import json

x = requests.get('https://api.scryfall.com/bulk-data')

bulk_data = x.json()



download_url = None

for item in bulk_data['data']:
    if item['type'] == 'default_cards':
        download_url = item['download_uri']
        break


if download_url:
    card_response = requests.get(download_url)

    file_path = "bulk_card_data.json"  
    with open(file_path, "w", encoding="utf-8") as json_file:
        json.dump(card_response.json(), json_file, indent=4)

