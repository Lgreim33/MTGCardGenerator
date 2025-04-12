import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBModel, DMatrix


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

def split_for_feature_coverage(df, feature_col, test_size=0.2, random_state=42):
    # Step 1: Flatten the feature into individual values with index mapping
    '''
    value_to_index = {}
    
    for idx, row in df.iterrows():
        for val in row[feature_col]:
            if val not in value_to_index:
                value_to_index[val] = idx  # only keep the first one
    '''
    # Step 2: Get the unique examples
    coverage_indices = list(df[feature_col])
    df_covered = df.loc[coverage_indices]

    # Step 3: Get the rest of the data
    df_remaining = df.drop(coverage_indices)

    # Step 4: Randomly split the remaining data
    df_train_rest, df_test = train_test_split(df_remaining, test_size=test_size, random_state=random_state)

    # Step 5: Combine guaranteed-coverage with train split
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



color_id_train, color_id_test = split_for_feature_coverage(textless,'color_identity',test_size=.2)

color_id_target = color_id_train['color_identity']
color_id_train.drop('color_identity',axis=1)


print(color_id_target)


model = XGBClassifier(enable_categorical=True)

#print(color_id_train.loc[:, color_id_train.columns != 'color_identity'])
model.fit(color_id_train,color_id_target)