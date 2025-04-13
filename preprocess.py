import os
import json
import pandas as pd
import re
import pickle

# Gets a dictionary of different mana symbols for easy matching during processing
def get_mana_symbols():

    fp = open("Data\ManaSymbols.json")

    data = json.load(fp)
    

    # Extract only the relevant symbols that are not "funny" this would indicate that the card is from an "un" set
    valid_mana_symbols = [
        symbol["symbol"]
        for symbol in data["data"]
        if symbol["represents_mana"]
    ]

    dict = {element: index for index, element in enumerate(valid_mana_symbols)}

    # Save dictionary
    with open(r"Dictionaries\ManaSymbolDict.pickle", "wb") as file:
        pickle.dump(dict, file)

    return dict

# Get as many possible keywords as we can and store them as a dictionaru
def get_keywords():
    fp_ab = open(r"Data\Types\keyword-abilities.json")
    fp_ac = open(r"Data\Types\keyword-actions.json")
    fp_aw = open(r"Data\Types\ability-words.json")
    data_ab = json.load(fp_ab)
    data_ac = json.load(fp_ac)
    data_aw = json.load(fp_aw)
    keywords = data_ab['data'] + data_ac['data'] + data_aw['data']

    dict = {element: index for index, element in enumerate(keywords)}

    # save the dictionary
    with open(r"Dictionaries\KeywordDict.pickle", "wb") as file:
        pickle.dump(dict, file)

    return dict

# Get all possible power symbols and save them in a Dictionary, returns the dict
def get_power_symbols():
    fp = open("Data\PowerSymbols.json")
    data = json.load(fp)
    powers = data['data']

    dict = {element: index for index, element in enumerate(powers)}

    # Save the dictionary
    with open(r"Dictionaries\PowerSymbolDict.pickle", "wb") as file:
        pickle.dump(dict, file)

    return dict
# Get all possible toughness symbols and save them in a Dictionary, returns the dict
def get_toughness_symbols():
    fp = open("Data\ToughnessSymbols.json")
    data = json.load(fp)
    toughness = data['data']

    dict = {element: index for index, element in enumerate(toughness)}

    # Save the dictionary
    with open(r"Dictionaries\ToughnessSymbolDict.pickle", "wb") as file:
        pickle.dump(dict, file)

    return dict

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

    dict = {element: index for index, element in enumerate(subtypes)}

    # Save the dictionary
    with open(r"Dictionaries\SubtypeDict.pickle", "wb") as file:
        pickle.dump(dict, file)

    # Return dictionary of {Subtype: Index}
    return dict
def pre_process(card_list):
    # Initialize dictionaries and constants
    subtypes = all_subtypes()
    mana_symbol_dict = get_mana_symbols()
    powers_dict = get_power_symbols()
    toughness_dict = get_toughness_symbols()
    keywords_dict = get_keywords()
    
    # Dictionary where the type matches the index at which the type can be represented by a one
    desired_types = {'Enchantment':0, 'Artifact':1, 'Creature':2, 'Instant':3, 'Sorcery':4, 'Kindred':5}
    color_identity_dict = {'W':0, 'U':1, 'B':2, 'R':3, 'G':4}

    with open(r"Dictionaries\TypeDict.pickle", "wb") as file:
        pickle.dump(desired_types, file)
    
    with open(r"Dictionaries\ColorDict.pickle", "wb") as file:
        pickle.dump(color_identity_dict, file)
    
    # Initialize lists to store the processed features for each card
    all_type_vectors = []
    all_subtype_vectors = []
    all_color_id_vecs = []
    all_mana_vectors = []
    all_keyword_vectors = []
    all_power_values = []
    all_toughness_values = []
    all_supertype_values = []  # List for supertype values (e.g., legendary)
    all_names = []
    all_rules = []

    
    # Process each card
    for card in card_list:
        type_vector = []
        subtype_vector = []
        mana_vector = []
        keyword_vector = []
        power_value = None
        toughness_value = None
        color_id_vec = [0 for _ in range(5)]
        supertype_value = None  # Initialize supertype_value to None

        try:
            # Process card types
            for type in card['types']:
                try:
                    type_vector.append(desired_types[type])
                except KeyError:
                    continue

            # Process card subtypes
            for type in card['subtypes']:
                try:
                    subtype_vector.append(subtypes[type])
                except KeyError:
                    continue

            # Process legendary supertype (optional)
            supertype_value = card.get('legendary', None)  # Using .get() to safely fetch 'legendary'

            # Process mana cost
            mana_cost = re.findall(r'\{[^}]+\}', card['mana_cost'])
            for cost in mana_cost:
                try:
                    mana_vector.append(mana_symbol_dict[cost])
                except KeyError:
                    continue

            # Process keywords
            for keyword in card['keywords']:
                try:
                    keyword_vector.append(keywords_dict[keyword])
                except KeyError:
                    continue

            # Process color identity (optional field, so no need to handle exception)
            for color in card['color_identity']:
                try:
                    color_id_vec[color_identity_dict[color]] = 1
                except KeyError:
                    continue

            # Process power and toughness (optional, can be None)
            try:
                power_value = powers_dict[card['power']]
            except KeyError:
                power_value = None

            try:
                toughness_value = toughness_dict[card['toughness']]
            except KeyError:
                toughness_value = None


            # Append all processed values (as tuples for hashability)
            all_type_vectors.append(tuple(type_vector))
            all_subtype_vectors.append(tuple(subtype_vector))
            all_color_id_vecs.append(tuple(color_id_vec))
            all_mana_vectors.append(tuple(mana_vector))
            all_keyword_vectors.append(tuple(keyword_vector))
            all_power_values.append(power_value)
            all_toughness_values.append(toughness_value)
            all_supertype_values.append(supertype_value)
            all_names.append(card['name'])
            all_rules.append(card['oracle_text'])


        except Exception as e:
            print(f"Skipping card {card.get('name', '<unknown>')} due to error: {e}")
            continue

    # Padding function — returns tuple for hashability
    def pad_features(features, max_len):
        return [tuple(f + (0,) * (max_len - len(f))) if len(f) < max_len else tuple(f[:max_len]) for f in features]

    # Determine max lengths and pad
    max_type_len = max(len(v) for v in all_type_vectors)
    max_subtype_len = max(len(v) for v in all_subtype_vectors)
    max_mana_len = max(len(v) for v in all_mana_vectors)
    max_keyword_len = max(len(v) for v in all_keyword_vectors)

    padded_type_vectors = pad_features(all_type_vectors, max_type_len)
    padded_subtype_vectors = pad_features(all_subtype_vectors, max_subtype_len)
    padded_mana_vectors = pad_features(all_mana_vectors, max_mana_len)
    padded_keyword_vectors = pad_features(all_keyword_vectors, max_keyword_len)
    
    # Create a DataFrame from the processed data
    df = pd.DataFrame({
        'name': all_names,
        'type_vector': tuple(padded_type_vectors),
        'subtype_vector': padded_subtype_vectors,
        'color_identity': all_color_id_vecs,
        'mana_vector': padded_mana_vectors,
        'keyword_vector': padded_keyword_vectors,
        'power_value': all_power_values,
        'toughness_value': all_toughness_values,
        'supertype_value': all_supertype_values,
        'rules_text' : all_rules   
    })
    
    return df
        

    
# While a card can be a couple of these types at once, we only care that the card is AT MOST these types
desired_types = ['Enchantment','Artifact','Creature','Instant','Sorcery','Kindred']

fp = open("Data\scryfall_bulk_data.json")
file = json.load(fp)

# Features we care about (We might include flavor text for fun)
features = ['name', 'color_identity', 'keywords', 'mana_cost','type_line','oracle_text','power','toughness']
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


card_dataframe = pre_process(card_dataset.values())



print(card_dataframe.head(5))

print(card_dataframe['type_vector'].value_counts())

# Save the dataframe
card_dataframe.to_pickle('card_dataframe.pkl')
