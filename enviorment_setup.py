import os

# Define the root structure
directories = [
    "Data",
    "Data/Types",
    "Dictionaries",
    "List_To_Singles",
    "Models",
    "SplitData/Color_ID",
    "SplitData/Keywords",
    "SplitData/Mana",
    "SplitData/Power",
    "SplitData/Subtype",
    "SplitData/Supertype",
    "SplitData/Toughness",
    "SplitData/Type",
]

# Create the directories
for path in directories:
    os.makedirs(path, exist_ok=True)

print("Directory structure created.")