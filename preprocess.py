import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor


def get_mana_symbols():

    fp = open("Data\ManaSymbols.json")

    data = json.load(fp)
    

    # Extract only the relevant symbols that are not "funny"
    valid_mana_symbols = [
        symbol["symbol"]
        for symbol in data["data"]
        if symbol["represents_mana"] and not symbol.get("funny", False)
    ]        
    print(valid_mana_symbols)


# Helper function to get all possible subtypes, retruns a dictionary of {Subtype: Index}
def all_subtypes():
    subtypes = set()
    # Read each file
    for file in os.listdir("Data\Types"):
        fp = open(f"Data\Types\{file}")
        file = json.load(fp)
        # Add each type to the set 
        for type in file['data']:
            subtypes.add(type)

    # Return dictionary of {Subtype: Index}
    return {element: index for index, element in enumerate(subtypes)}
def pre_process(card_list):

    # Get subtype dict
    subtypes = all_subtypes()

    mana = get_mana_symbols()
   
    # Dictionary where the type matches the index at which the type can be represented by a one
    desired_types = {'Enchantment':0,'Artifact':1,'Creature':2,'Instant':3,'Sorcery':4}
    for card in card_list:
        type_vector = np.zeros(5)
        subtype_vector = np.zeros(len(subtypes))
        for type in card['types']:
            
            type_vector[desired_types[type]] = 1
        
        for type in card['subtypes']:
            
            subtype_vector[subtypes[type]] = 1


        # Just return this for now for testing
        return type_vector, subtype_vector
        

    
# While a card can be a couple of these types at once, we only care that the card is AT MOST these types
desired_types = ['Enchantment','Artifact','Creature','Instant','Sorcery']

fp = open("Data\scryfall_bulk_data.json")
file = json.load(fp)

# Features we care about (We might include flavor text for fun)
features = ['name','colors', 'color_identity', 'keywords', 'mana_cost','cmc','type_line','oracle_text','power','toughness']
card_dataset = dict()


# Helper function to split type_line into components
def split_type_line(type_line):
    # Check if 'Legendary' is part of the type line
    legendary = 1 if 'Legendary' in type_line else 0
    
    # Split by ' — ' to separate the types and subtypes
    parts = type_line.split(' — ')
    types = (parts[0].replace("Legendary","")).split()  # The types before the dash
    subtypes = parts[1].split() if len(parts) > 1 else []  # The subtypes after the dash
    
    return legendary, types, subtypes

# Iterate through bulk data
for card in file:
    # If the card is of a desired type, get its desired features
    if 'type_line' in card and any(t in card['type_line'] for t in desired_types):
        # Split the type_line into Legendary, Types, and Subtypes
        legendary, types, subtypes = split_type_line(card['type_line'])
        
        # Add to the dataset
        card_dataset[card['name']] = {
            **{feature: card.get(feature, None) for feature in features if not feature =='type_line'},  # Add other features, besides type_line
            'legendary': legendary,  
            'types': types,          
            'subtypes': subtypes     
        }
        
            
# Print Size of dataset
print(len(card_dataset))

# Test
print(card_dataset['Sliver Hivelord'])

type_vector, subtype_vector = pre_process([card_dataset['Copper Myr']])

print(type_vector)
print(subtype_vector)


