import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor

def pre_process(card_list):
    for card in card_list:


        return
        

    
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
    legendary = 'Legendary' in type_line
    
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


