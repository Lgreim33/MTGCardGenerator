import requests
import json

features = ['card-types', 'artifact-types', 'creature-types','enchantment-types','spell-types','supertypes','keyword-abilities']
data_dict = {}

modular_request = "https://api.scryfall.com/catalog/"

for feature in features:
    response = requests.get(modular_request + feature)
    
    if response.status_code == 200:
        data = response.json()
        
        # Save each response in its own JSON file
        filename = f"Data\{feature}.json"
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)
        
        print(f"Saved {filename}")
    else:
        print(f"Failed to fetch {feature}, status code: {response.status_code}")


# Scryfall API endpoint for mana symbols
url = "https://api.scryfall.com/symbology"

# Fetch data from Scryfall
response = requests.get(url)
data = response.json()

# Extract only the relevant symbols that are not "funny"
valid_mana_symbols = [
    symbol["symbol"]
    for symbol in data["data"]
    if symbol["represents_mana"] and not symbol.get("funny", False)
]

# Print or save the results
print(valid_mana_symbols)
filename = f"Data\ManaSymbols.json"
with open(filename, "w", encoding="utf-8") as file:
    json.dump(data, file, indent=4)