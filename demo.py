import joblib
import pandas as pd
import pickle
import re
import json
import numpy as np
from text_gen import TextGen

# Demo class to contain prediction pipeline
class Demo:

    def __init__(self):
        # Load models
        Color_ID_Model = joblib.load('Models/color_id_model.pkl')
        Mana_Cost_Model = joblib.load('Models/mana_model.pkl')
        Type_Model =joblib.load('Models/type_model.pkl')
        Subtype_Model = joblib.load('Models/subtype_model.pkl')
        Supertype_Model =joblib.load('Models/supertype_model.pkl')
        Keyword_Model = joblib.load('Models/keyword_model.pkl')
        Power_Model = joblib.load('Models/power_model.pkl')
        Toughness_Model = joblib.load('Models/toughness_model.pkl')

        # Dictionary of models for easy lookup
        self.model_dict = {'color_identity' : Color_ID_Model,
                    'keyword_vector': Keyword_Model,
                        'mana_vector': Mana_Cost_Model,
                        'type_vector': Type_Model,
                            'subtype_vector': Subtype_Model,
                            'power_value': Power_Model,
                                'toughness_value': Toughness_Model,
                                'supertype_value': Supertype_Model}

        # Load Dictionaries to translate the passed data
        with open("Dictionaries/ColorDict.pickle", "rb") as f:
            self.color_dict = pickle.load(f)

        with open("Dictionaries/ManaSymbolDict.pickle", "rb") as f:
            self.mana_dict = pickle.load(f)

        with open("Dictionaries/KeywordDict.pickle", "rb") as f:
            self.keyword_dict = pickle.load(f)

        with open("Dictionaries/PowerSymbolDict.pickle", "rb") as f:
            self.power_dict = pickle.load(f)

        with open("Dictionaries/SubtypeDict.pickle", "rb") as f:
            self.subtype_dict = pickle.load(f)

        with open("Dictionaries/ToughnessSymbolDict.pickle", "rb") as f:
            self.toughness_dict = pickle.load(f)

        with open("Dictionaries/TypeDict.pickle", "rb") as f:
            self.type_dict = pickle.load(f)
                
        # Store in dictionary for easy matching with sample and dictionary
        self.symbol_to_vector_dicts = {
            'color_identity': self.color_dict,
            'keyword_vector': self.keyword_dict,
            'mana_vector': self.mana_dict,
            'type_vector': self.type_dict,
            'subtype_vector': self.subtype_dict,
            'power_value': self.power_dict,
            'toughness_value': self.toughness_dict,
            'supertype_value': {0: False, 1: True} 
            }

        # Load Dictionaries to translate vectors to singleton values
        with open("List_To_Singles/keyword_dict.pkl", "rb") as f:
            self.single_keyword_dict = pickle.load(f)

        with open("List_To_Singles/mana_cost_dict.pkl", "rb") as f:
            self.single_mana_dict = pickle.load(f)

        with open("List_To_Singles/subtype_dict.pkl", "rb") as f:
            self.single_subtype_dict = pickle.load(f)

        with open("List_To_Singles/type_dict.pkl", "rb") as f:
            self.single_type_dict = pickle.load(f)

        with open("List_To_Singles/color_id_dict.pkl", "rb") as f:
            self.single_color_dict = pickle.load(f)

        self.vector_to_singleton_dicts = {
            'color_identity': self.single_color_dict,
            'keyword_vector': self.single_keyword_dict,
            'mana_vector': self.single_mana_dict,
            'type_vector': self.single_type_dict,
            'subtype_vector': self.single_subtype_dict,
        }

        # Create inverse dictionaries, these will be for retranslating the model output
        self.inv_type_dict = {v: k for k, v in self.type_dict.items()}
        self.inv_subtype_dict = {v: k for k, v in self.subtype_dict.items()}
        self.inv_color_dict = {v: k for k, v in self.color_dict.items()}
        self.inv_mana_dict = {v: k for k, v in self.mana_dict.items()}
        self.inv_keyword_dict = {v: k for k, v in self.keyword_dict.items()}
        self.inv_power_dict = {v: k for k, v in self.power_dict.items()}
        self.inv_toughness_dict = {v: k for k, v in self.toughness_dict.items()}

        self.vector_to_symbol_dicts = {
            'color_identity': self.inv_color_dict,
            'keyword_vector': self.inv_keyword_dict,
            'mana_vector': self.inv_mana_dict,
            'type_vector': self.inv_type_dict,
            'subtype_vector': self.inv_subtype_dict,
            'power_value': self.inv_power_dict,
            'toughness_value': self.inv_toughness_dict,
            'supertype_value': {False: 0, True: 1}
        }

        self.inv_single_keyword_dict = {v: k for k, v in self.single_keyword_dict.items()}
        self.inv_single_mana_dict = {v: k for k, v in self.single_mana_dict.items()}
        self.inv_single_subtype_dict = {v: k for k, v in self.single_subtype_dict.items()}
        self.inv_single_type_dict = {v: k for k, v in self.single_type_dict.items()}
        self.inv_single_color_dict = {v: k for k, v in self.single_color_dict.items()}

        self.singleton_to_vector_dicts = {
            'color_identity': self.inv_single_color_dict,
            'keyword_vector': self.inv_single_keyword_dict,
            'mana_vector': self.inv_single_mana_dict,
            'type_vector': self.inv_single_type_dict,
            'subtype_vector': self.inv_single_subtype_dict,
        }
    
    def pad_feature(self,feature, max_len):
        return tuple(feature + (0,) * (max_len - len(feature))) if len(feature) < max_len else tuple(feature[:max_len])
    

    # Process the sample for model prediction
    def process(self,sample):

        # Copy the input to avoid mutating the original
        processed = sample.copy()
        fp = open("max_vector_lengths.json")
        max_feature_lens = json.load(fp)

        # Convert each feature into the correct vector format
        if sample['color_identity'] is not None:
            vec = tuple(self.keyword_dict[kw] for kw in sample['color_identity'] if kw in self.keyword_dict)
            processed['color_identity'] = self.pad_feature(vec,int(max_feature_lens['color_identity']))
            
            

        if sample['keyword_vector'] is not None:
            vec = tuple(self.keyword_dict[kw] for kw in sample['keyword_vector'] if kw in self.keyword_dict)
            processed['keyword_vector'] = self.pad_feature(vec,int(max_feature_lens['keyword_vector']))


        if sample['mana_vector'] is not None:
            mana_symbols = re.findall(r'\{[^}]+\}', sample['mana_vector'])
            vec = tuple(self.mana_dict[sym] for sym in mana_symbols if sym in self.mana_dict)
            processed['mana_vector'] = self.pad_feature(vec,int(max_feature_lens['mana_vector']))

        if sample['type_vector'] is not None:
            vec = tuple(self.type_dict[t] for t in sample['type_vector'] if t in self.type_dict)
            processed['type_vector'] = self.pad_feature(vec,int(max_feature_lens['type_vector']))

        if sample['subtype_vector'] is not None:
            vec = tuple(self.subtype_dict[st] for st in sample['subtype_vector'] if st in self.subtype_dict)
            processed['subtype_vector'] = self.pad_feature(vec,int(max_feature_lens['subtype_vector']))

        if sample['power_value'] is not None:
            processed['power_value'] = self.power_dict.get(sample['power_value'], max(self.power_dict.values()))

        if sample['toughness_value'] is not None:
            processed['toughness_value'] = self.toughness_dict.get(sample['toughness_value'], max(self.toughness_dict.values()))

        if sample['supertype_value'] is not None:
            processed['supertype_value'] = sample['supertype_value']


        

        return processed
    

    def convert_vectors_to_singletons(self, processed_sample):

        encoded_sample = processed_sample.copy()
        for key in processed_sample:
            if key in self.vector_to_singleton_dicts:
                vector = processed_sample[key]
                singleton_value = self.vector_to_singleton_dicts[key].get(vector)
                if singleton_value is not None:
                    encoded_sample[key] = singleton_value
                    
                else:
                    # Handle unseen combinations
                    continue
                    #processed_sample[key]
        return encoded_sample
    
    def decode(self, encoded_sample):
        
        decoded_sample = encoded_sample.copy()
        
        for key,value in encoded_sample.items():
            if key in self.singleton_to_vector_dicts.keys():
                symbol_keys = self.singleton_to_vector_dicts[key][value]
                symbol_list = []
                for symbol in symbol_keys:
                    # Zero indicates we have run out of categories
                    if symbol == 0:
                        break
                    symbol_list.append(self.vector_to_symbol_dicts[key][symbol])

                decoded_sample[key] = symbol_list

            else:
                decoded_sample[key] = self.vector_to_symbol_dicts[key][value]

        return decoded_sample


    def predict(self,sample,random_sample=False):

        processed_sample = self.process(sample)
        encoded_sample = self.convert_vectors_to_singletons(processed_sample)


        for key,value in encoded_sample.items():
            if value is None:
                encoded_sample[key] = -1
                

        for key in encoded_sample:
            if (isinstance(encoded_sample[key], int) and encoded_sample[key] == -1):
                # Prepare the input for the model — remove the target feature
                model_input = {k: v for k, v in encoded_sample.items() if k != key}

                # Convert model input to DataFrame
                df_input = pd.DataFrame([model_input])

                # Predict the missing feature using the corresponding model
                model = self.model_dict[key]
                
                if random_sample:
                    probas = model.predict_proba(df_input)[0]
                    top3_indices = np.argsort(probas)[-2:][::-1]  # top 2 indices
                    predicted_value = np.random.choice(top3_indices)
                else:
                    predicted_value = model.predict(df_input)[0]

                # Update the encoded sample with the prediction
                encoded_sample[key] = predicted_value


        decoded_sample = self.decode(encoded_sample)

        return decoded_sample



demo = Demo()


# Example of potential input with missing values to be predicted
sample = {
    'type_vector': ['Enchantment','Creature'],
    'subtype_vector': None,
    'color_identity': None,
    'mana_vector': None,
    'keyword_vector': None,
    'power_value': None,
    'toughness_value': '1',
    'supertype_value': 1,
}

# Get Predicition
prediction = demo.predict(sample,random_sample=True)
print(prediction)

# Give option to run for text generation in case they dont have an API key
while(True):
    run_text = str(input("Would you like to run text generation? You will need a hugging face API key to download the model (y/n)"))

    if(run_text == "y"):

        # Instantiate Llama3 Model
        text_generator = TextGen()
        '''
        Expected Format:
        def generate_card(
            type: list,
            subtypes: list,
            color_identity: list,
            mana_cost: str,
            power: str,
            toughness: str,
            keywords: list,
            supertype: bool
            max_tokens: int = 250
        )'''
        
        # generate text
        generated_card = text_generator.generate_card(prediction['type_vector'],
                                    prediction['subtype_vector'],
                                    prediction['color_identity'],
                                    prediction['mana_vector'],
                                    prediction['power_value'],
                                    prediction['toughness_value'],
                                    prediction['keyword_vector'],
                                    prediction['supertype_value'],
                                    max_tokens=1000)
        
        print(generated_card['name'])
        print(generated_card['card_display'])
        print(generated_card['rules_text'])

    else:
        break