import pickle
import pandas as pd
from sklearn.model_selection import train_test_split


with open("card_dataframe.pkl","rb") as file:
    card_dataframe = pd.DataFrame(pickle.load(file))


print(card_dataframe.head())


# We need to train a model for each target, so we're going to split the data in a way that the model can see all possible entries


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

def split_for_feature_coverage(df, feature_col, test_size=0.2, random_state=42):
    # Step 1: Flatten the feature into individual values with index mapping
    value_to_index = {}
    
    for idx, row in df.iterrows():
        for val in row[feature_col]:
            if val not in value_to_index:
                value_to_index[val] = idx  # only keep the first one
    
    # Step 2: Get the unique examples
    coverage_indices = list(value_to_index.values())
    df_covered = df.loc[coverage_indices]

    # Step 3: Get the rest of the data
    df_remaining = df.drop(coverage_indices)

    # Step 4: Randomly split the remaining data
    df_train_rest, df_test = train_test_split(df_remaining, test_size=test_size, random_state=random_state)

    # Step 5: Combine guaranteed-coverage with train split
    df_train = pd.concat([df_covered, df_train_rest]).reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_test


color_id_train, color_id_test = split_for_feature_coverage(card_dataframe,'color_identity',test_size=.2)



print(len(color_id_train['color_identity'].value_counts()))