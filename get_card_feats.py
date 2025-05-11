import requests
import json

# features to get
features = ['card-types', 'artifact-types', 'creature-types','enchantment-types','spell-types','supertypes','keyword-abilities' ,'keyword-actions','ability-words']
data_dict = {}

modular_request = "https://api.scryfall.com/catalog/"

for feature in features:
    response = requests.get(modular_request + feature)
    
    if response.status_code == 200:
        data = response.json()
        
        # Save each response in its own JSON file
        filename = f"Data\Types\{feature}.json"
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        
        print(f"Saved {filename}")
    else:
        print(f"Failed to fetch {feature}, status code: {response.status_code}")


# Get Mana Symbols

# Scryfall API endpoint for mana symbols
url_mana = "https://api.scryfall.com/symbology"

# Fetch data from Scryfall
response = requests.get(url_mana)
data = response.json()



filename = f"Data\ManaSymbols.json"
with open(filename, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)


url_toughness = "https://api.scryfall.com/catalog/toughnesses"


# Fetch data from Scryfall
response = requests.get(url_toughness)
data = response.json()



filename = f"Data\ToughnessSymbols.json"
with open(filename, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)

url_powers = "https://api.scryfall.com/catalog/powers"

# Fetch data from Scryfall
response = requests.get(url_toughness)
data = response.json()

filename = f"Data\PowerSymbols.json"
with open(filename, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)




