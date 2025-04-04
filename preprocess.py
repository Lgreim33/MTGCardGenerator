import os
import json
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
import re


def get_mana_symbols():

    fp = open("Data\ManaSymbols.json")

    data = json.load(fp)
    

    # Extract only the relevant symbols that are not "funny" this would indicate that the card is from an "un" set
    valid_mana_symbols = [
        symbol["symbol"]
        for symbol in data["data"]
        if symbol["represents_mana"]
    ]        
    return {element: index for index, element in enumerate(valid_mana_symbols)}

def get_types():
    fp = open("Data\keyword-abilities.json")
    data = json.load(fp)
    keywords = data['data']

    return {element: index for index, element in enumerate(keywords)}

def get_power_symbols():


    return

def get_toughness_symbols():


    return

# Helper function to get all possible subtypes, retruns a dictionary of {Subtype: Index}
def all_subtypes():
    subtypes = list()
    # Read each file
    for file in os.listdir("Data\Types"):
        fp = open(f"Data\Types\{file}")
        file = json.load(fp)
        # Add each type to the set 
        try:
            for type in file['data']:
                subtypes.append(type)
        except KeyError:
            continue

    # Return dictionary of {Subtype: Index}
    return {element: index for index, element in enumerate(subtypes)}
def pre_process(card_list):

    # Get subtype dict
    subtypes = all_subtypes()

    mana = get_mana_symbols()
   
    # Dictionary where the type matches the index at which the type can be represented by a one
    desired_types = {'Enchantment':0,'Artifact':1,'Creature':2,'Instant':3,'Sorcery':4 , 'Kindred':5}
    for card in card_list:
        type_vector = []
        subtype_vector = []
        mana_vector = []
        for type in card['types']:

            try:
            
                type_vector.append(desired_types[type])
            except KeyError:
                continue
        for type in card['subtypes']:
            try:
                subtype_vector.append(subtypes[type])
            except KeyError:
                continue
        # Mana cost is represented as a string, so we should transform it into a list
        mana_cost = re.findall(r'\{[^}]+\}', card['mana_cost'])
        for cost in mana_cost:
            mana_vector.append(mana[cost])



        # Just return this for now for testing
        return np.array(type_vector), np.array(subtype_vector), np.array(mana_vector)
        

    
# While a card can be a couple of these types at once, we only care that the card is AT MOST these types
desired_types = ['Enchantment','Artifact','Creature','Instant','Sorcery','Kindred']

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
    # If the card is of a desired type, get its features
    if 'type_line' in card: 
        # Split the type_line into Legendary, Types, and Subtypes
        legendary, types, subtypes = split_type_line(card['type_line'])
        
        if set(types).issubset(set(desired_types)) and "//" not in card['name']:

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
print(card_dataset['All Will Be One'])
i = 0
for card in card_dataset:

    try:
        type_vector, subtype_vector, mana_vector = pre_process([card_dataset[card]])
        #print(card)
        i+=1

    except Exception:
        continue

print(i)
#print(type_vector)
#print(subtype_vector)
#print(mana_vector)


