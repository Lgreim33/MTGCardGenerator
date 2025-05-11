import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Function to plot the distribution of all target variables as subplots in a single figure
def plot_target_distributions_subplots(df, feature_cols, threshold_percentage=1.0):

    # Number of features to plot
    num_features = len(feature_cols)
    
    # Set up the subplot grid
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten() 

    # Plot each feature's distribution in a subplot
    for idx, feature_name in enumerate(feature_cols):
        target_series = df[feature_name]
        ax = axes[idx]

        # Calculate class counts
        class_counts = target_series.value_counts()
        total_count = len(target_series)
        threshold_count = (threshold_percentage / 100) * total_count

        # Group small classes into 'Other'
        small_classes = class_counts[class_counts < threshold_count]
        other_count = small_classes.sum()  # Total count of 'Other' instances
        num_other_classes = len(small_classes)  # Number of classes grouped into 'Other'

        # Filter out small classes and add 'Other'
        large_classes = class_counts[class_counts >= threshold_count]
        if other_count > 0:
            large_classes = pd.concat([
                large_classes,
                pd.Series(other_count, index=['Other'])
            ])

        # Create labels and plot the pie chart
        labels = [str(label) if label != 'Other' else f'Other ({num_other_classes} classes)' for label in large_classes.index]
        ax.pie(large_classes, labels=labels, autopct='%1.1f%%', startangle=140)
        ax.set_title(f'{feature_name}\n(Total Classes: {len(class_counts)})')
        ax.axis('equal')  # Equal aspect ratio ensures the pie chart is circular

    # Turn off any unused subplots (if num_features < rows * cols)
    for idx in range(num_features, len(axes)):
        axes[idx].axis('off')

    # Adjust layout and save the plot
    plt.suptitle(f'Distribution of Target Variables (Threshold: {threshold_percentage}%)', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('target_distributions_subplots.png')
    plt.close()



# Read the stored data file
with open("card_dataframe.pkl","rb") as file:
    card_dataframe = pd.DataFrame(pickle.load(file))



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

    # Get the unique examples
    coverage_indices = df.groupby(feature_col).apply(lambda x: x.sample(1, random_state=random_state)).index.get_level_values(1)
    df_covered = df.loc[coverage_indices]

    # Get the rest of the data
    df_remaining = df.drop(coverage_indices)


    # Randomly split the remaining data
    df_train_rest, df_test = train_test_split(df_remaining, test_size=test_size, random_state=random_state)

    # Combine guaranteed-coverage with train split
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


with open("List_to_singles/type_dict.pkl", "wb") as f:
    pickle.dump(type_dict, f)
with open("List_to_singles/subtype_dict.pkl", "wb") as f:
    pickle.dump(subtype_dict, f)
with open("List_to_singles/color_id_dict.pkl", "wb") as f:
    pickle.dump(color_id_dict, f)
with open("List_to_singles/mana_cost_dict.pkl", "wb") as f:
    pickle.dump(mana_cost_dict, f)
with open("List_to_singles/keyword_dict.pkl", "wb") as f:
    pickle.dump(keyword_dict, f)

textless['type_vector'] = textless['type_vector'].map(type_dict)
textless['subtype_vector'] = textless['subtype_vector'].map(subtype_dict)
textless['color_identity'] = textless['color_identity'].map(color_id_dict)
textless['mana_vector'] = textless['mana_vector'].map(mana_cost_dict)
textless['keyword_vector'] = textless['keyword_vector'].map(keyword_dict)


# Define the list of target features to plot
feature_cols = [
    'color_identity',
    'type_vector',
    'supertype_value',
    'subtype_vector',
    'mana_vector',
    'keyword_vector',
    'power_value',
    'toughness_value'
]

# Plot the pre-split distributions as subplots
plot_target_distributions_subplots(textless, feature_cols)


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

# Mana split
mana_train, mana_test = split_for_feature_coverage(textless,'mana_vector',test_size=.2)

mana_train_target = mana_train['mana_vector']
mana_test_target = mana_test['mana_vector']
mana_train = mana_train.drop('mana_vector',axis=1)
mana_test = mana_test.drop('mana_vector',axis=1)


mana_test.to_pickle("SplitData/Mana/mana_test.pkl")
mana_train_target.to_pickle("SplitData/Mana/mana_train_target.pkl")
mana_test_target.to_pickle("SplitData/Mana/mana_test_target.pkl")
mana_train.to_pickle("SplitData/Mana/mana_train.pkl")


#keyword split
keyword_train, keyword_test = split_for_feature_coverage(textless,'keyword_vector',test_size=.2)

keyword_train_target = keyword_train['keyword_vector']
keyword_test_target = keyword_test['keyword_vector']
keyword_train = keyword_train.drop('keyword_vector',axis=1)
keyword_test = keyword_test.drop('keyword_vector',axis=1)

keyword_test.to_pickle("SplitData/Keywords/keyword_test.pkl")
keyword_train_target.to_pickle("SplitData/Keywords/keyword_train_target.pkl")
keyword_test_target.to_pickle("SplitData/Keywords/keyword_test_target.pkl")
keyword_train.to_pickle("SplitData/Keywords/keyword_train.pkl")

# Power Split
power_train, power_test = split_for_feature_coverage(textless,'power_value',test_size=.2)

power_train_target = power_train['power_value'].astype(int)
power_test_target = power_test['power_value'].astype(int)
power_train = power_train.drop('power_value',axis=1)
power_test = power_test.drop('power_value',axis=1)

power_test.to_pickle("SplitData/Power/power_test.pkl")
power_train_target.to_pickle("SplitData/Power/power_train_target.pkl")
power_test_target.to_pickle("SplitData/Power/power_test_target.pkl")
power_train.to_pickle("SplitData/Power/power_train.pkl")

# toughness split
toughness_train, toughness_test = split_for_feature_coverage(textless,'toughness_value',test_size=.2)

toughness_train_target = toughness_train['toughness_value'].astype(int)
toughness_test_target = toughness_test['toughness_value'].astype(int)
toughness_train = toughness_train.drop('toughness_value',axis=1)
toughness_test = toughness_test.drop('toughness_value',axis=1)

toughness_test.to_pickle("SplitData/Toughness/toughness_test.pkl")
toughness_train_target.to_pickle("SplitData/Toughness/toughness_train_target.pkl")
toughness_test_target.to_pickle("SplitData/Toughness/toughness_test_target.pkl")
toughness_train.to_pickle("SplitData/Toughness/toughness_train.pkl")
