import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split



# Read the stored data file
with open("card_dataframe.pkl","rb") as file:
    card_dataframe = pd.DataFrame(pickle.load(file))


print(card_dataframe.head())


# We need to train a model for each target, so we're going to split the data in a way that the model can see all possible entries
'''
We also need to encode our vectors into unique scalar values, as XGboost cannot handle lists/objects as input
later on we will still be able to transform these labels into our vector representation, and from there we can 
derive actual labels. Its somewhat unfortunate because it means we wont see non-existent combinations, but for the scope of this project
its necessary to represent the data this way.

Vector Type Features:
type_vector
subtype_vector
mana_vector
color_identity
keyword_vector
'''


'''
# uncomment this to get a look at the data distribution, commented out to avoid bulky terminal prints at each run

# View the data split
print("Types:")
print(card_dataframe['type_vector'].value_counts())

print("Subtypes:")
print(card_dataframe['subtype_vector'].value_counts())

print("Mana Value:")
print(card_dataframe['mana_vector'].value_counts())

print("Supertypes:")
print(card_dataframe['supertype_value'].value_counts())

print("Color Identity:")
print(len(card_dataframe['color_identity'].value_counts()))

print("Power:")
print(card_dataframe['power_value'].value_counts())

print("Toughness:")
print(card_dataframe['toughness_value'].value_counts())


print("Description of Data:")
print(card_dataframe['power_value'].describe())

'''
def split_for_feature_coverage(df, feature_col, test_size=0.2, random_state=42):

    # Step 1: Get the unique examples
    coverage_indices = df.groupby(feature_col).apply(lambda x: x.sample(1, random_state=random_state)).index.get_level_values(1)
    df_covered = df.loc[coverage_indices]

    print(df_covered.value_counts())
    # Step 2: Get the rest of the data
    df_remaining = df.drop(coverage_indices)


    # Step 3: Randomly split the remaining data
    df_train_rest, df_test = train_test_split(df_remaining, test_size=test_size, random_state=random_state)

    # Step 4: Combine guaranteed-coverage with train split
    df_train = pd.concat([df_covered, df_train_rest]).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)
    

    return df_train, df_test
# We want classification to be agnostic to the name and text
textless = card_dataframe.loc[:, (card_dataframe.columns != 'name')]
textless = textless.loc[:, (textless.columns != 'rules_text')]



# Convert each vector type feature into a single value
type_dict = {element: index for index, element in enumerate(textless['type_vector'].unique())}
subtype_dict = {element: index for index, element in enumerate(textless['subtype_vector'].unique())}
color_id_dict = {element: index for index, element in enumerate(textless['color_identity'].unique())}
mana_cost_dict = {element: index for index, element in enumerate(textless['mana_vector'].unique())}
keyword_dict = {element: index for index, element in enumerate(textless['keyword_vector'].unique())}


textless['type_vector'] = textless['type_vector'].map(type_dict)
textless['subtype_vector'] = textless['subtype_vector'].map(subtype_dict)
textless['color_identity'] = textless['color_identity'].map(color_id_dict)
textless['mana_vector'] = textless['mana_vector'].map(mana_cost_dict)
textless['keyword_vector'] = textless['keyword_vector'].map(keyword_dict)


# Color identity training split
color_id_train, color_id_test = split_for_feature_coverage(textless,'color_identity',test_size=.2)

color_id_train_target = color_id_train['color_identity']
color_id_test_target = color_id_test['color_identity']
color_id_train=color_id_train.drop('color_identity',axis=1)
color_id_test=color_id_test.drop('color_identity',axis=1)



color_id_test.to_pickle("SplitData/Color_ID/color_id_test.pkl")
color_id_train_target.to_pickle("SplitData/Color_ID/color_id_train_target.pkl")
color_id_test_target.to_pickle("SplitData/Color_ID/color_id_test_target.pkl")
color_id_train.to_pickle("SplitData/Color_ID/color_id_train.pkl")


# Type training split
type_train, type_test = split_for_feature_coverage(textless,'type_vector',test_size=.2)

type_train_target = type_train['type_vector']
type_test_target = type_test['type_vector']
type_train = type_train.drop('type_vector',axis=1)
type_test = type_test.drop('type_vector',axis=1)

type_test.to_pickle("SplitData/Type/type_test.pkl")
type_train_target.to_pickle("SplitData/Type/type_train_target.pkl")
type_test_target.to_pickle("SplitData/Type/type_test_target.pkl")
type_train.to_pickle("SplitData/Type/type_train.pkl")


# Supertype split
supertype_train, supertype_test = split_for_feature_coverage(textless,'supertype_value',test_size=.2)

supertype_train_target = supertype_train['supertype_value']
supertype_test_target = supertype_test['supertype_value']
supertype_train = supertype_train.drop('supertype_value',axis=1)
supertype_test = supertype_test.drop('supertype_value',axis=1)

supertype_test.to_pickle("SplitData/Supertype/supertype_test.pkl")
supertype_train_target.to_pickle("SplitData/Supertype/supertype_train_target.pkl")
supertype_test_target.to_pickle("SplitData/Supertype/supertype_test_target.pkl")
supertype_train.to_pickle("SplitData/Supertype/supertype_train.pkl")

# Subtype training split
subtype_train, subtype_test = split_for_feature_coverage(textless,'subtype_vector',test_size=.2)

subtype_train_target = subtype_train['subtype_vector']
subtype_test_target = subtype_test['subtype_vector']
subtype_train = subtype_train.drop('subtype_vector',axis=1)
subtype_test = subtype_test.drop('subtype_vector',axis=1)

subtype_test.to_pickle("SplitData/Subtype/subtype_test.pkl")
subtype_train_target.to_pickle("SplitData/Subtype/subtype_train_target.pkl")
subtype_test_target.to_pickle("SplitData/Subtype/subtype_test_target.pkl")
subtype_train.to_pickle("SplitData/Subtype/subtype_train.pkl")

mana_train, mana_test = split_for_feature_coverage(textless,'mana_vector',test_size=.2)

mana_train_target = mana_train['mana_vector']
mana_test_target = mana_test['mana_vector']
mana_train = mana_train.drop('mana_vector',axis=1)
mana_test = mana_test.drop('mana_vector',axis=1)

mana_test.to_pickle("SplitData/Mana/mana_test.pkl")
type_train_target.to_pickle("SplitData/Mana/mana_train_target.pkl")
type_test_target.to_pickle("SplitData/Mana/mana_test_target.pkl")
type_train.to_pickle("SplitData/Mana/mana_train.pkl")


keyword_train, keyword_test = split_for_feature_coverage(textless,'keyword_vector',test_size=.2)

keyword_train_target = keyword_train['keyword_vector']
keyword_test_target = keyword_test['keyword_vector']
keyword_train = keyword_train.drop('keyword_vector',axis=1)
keyword_test = keyword_test.drop('keyword_vector',axis=1)

keyword_test.to_pickle("SplitData/Keywords/keyword_test.pkl")
keyword_train_target.to_pickle("SplitData/Keywords/keyword_train_target.pkl")
keyword_test_target.to_pickle("SplitData/Keywords/keyword_test_target.pkl")
keyword_train.to_pickle("SplitData/Keywords/keyword_train.pkl")

power_train, power_test = split_for_feature_coverage(textless,'power_value',test_size=.2)

print(set(power_train['power_value']))
power_train_target = power_train['power_value'].astype(int)
print(set(power_train_target))
power_test_target = power_test['power_value'].astype(int)
power_train = power_train.drop('power_value',axis=1)
power_test = power_test.drop('power_value',axis=1)

power_test.to_pickle("SplitData/Power/power_test.pkl")
power_train_target.to_pickle("SplitData/Power/power_train_target.pkl")
power_test_target.to_pickle("SplitData/Power/power_test_target.pkl")
power_train.to_pickle("SplitData/Power/power_train.pkl")


toughness_train, toughness_test = split_for_feature_coverage(textless,'toughness_value',test_size=.2)

toughness_train_target = toughness_train['toughness_value'].astype(int)
toughness_test_target = toughness_test['toughness_value'].astype(int)
toughness_train = toughness_train.drop('toughness_value',axis=1)
toughness_test = toughness_test.drop('toughness_value',axis=1)

toughness_test.to_pickle("SplitData/Toughness/toughness_test.pkl")
toughness_train_target.to_pickle("SplitData/Toughness/toughness_train_target.pkl")
toughness_test_target.to_pickle("SplitData/Toughness/toughness_test_target.pkl")
toughness_train.to_pickle("SplitData/Toughness/toughness_train.pkl")
