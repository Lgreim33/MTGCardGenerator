# Downloading Model 
## Huggingface Authentication 
from huggingface_hub import login
from dotenv import load_dotenv
import os
from huggingface_hub import hf_hub_download
from llama_cpp import Llama



class TextGen:

    def __init__(self):
        load_dotenv()

        login(token=os.environ.get("HUGGINGFACE_TOKEN"))


        model_path = hf_hub_download(
            repo_id="QuantFactory/Meta-Llama-3-8B-Instruct-GGUF",
            filename="Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
        )

        print(f"Model downloaded to: {model_path}")

        ## Loading model into memory 
    

        self.llama3_mtg = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=4,
            use_mlock=False
        )

    # Card Generation Creation 

    def generate_card(self,
        type: list,
        subtypes: list,
        color_identity: list,
        mana_cost: str,
        power: str,
        toughness: str,
        keywords: list,
        Legendary:bool,
        max_tokens: int = 250
    ):
        
        prompt = f"""Create a Magic: The Gathering card using the following characteristics:

    - Type: {', '.join(type)}
    - Subtypes: {', '.join(subtypes)}
    - Color Identity: {', '.join(color_identity)}
    - Mana Cost: {mana_cost}
    - Power/Toughness: {power}/{toughness}
    - Keywords: {', '.join(keywords)}
    - Legendary: {Legendary}

    Every feature should influence the card's rules text
    
    Return the card in the following format:

    Name: <The card's name>,
    Rules Text: <Describes the card’s abilities, effects, and how it functions during gameplay. Use formal Magic: The Gathering rules text. Include triggers, costs, and any conditions needed.>
    """

        result = self.llama3_mtg(prompt, max_tokens=max_tokens)
        text = result["choices"][0]["text"].strip()

        # Ensure it starts with Rules Text for consistency
        text = f"Rules Text: {text}"

        # Parse model response
        card = {}
        for line in text.splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                card[key.strip()] = value.strip()

        # Fallbacks
        fallback_name = next((line for line in text.splitlines() if line.strip()), "Unnamed")
        card_name = card.get("Name", fallback_name)
        final_rules = card.get("Rules Text", "[No rules text generated]")

        # Display
        card_display = f"""\
    Name: {card_name}
    Type: {type}
    Subtypes: {', '.join(subtypes)}
    Color Identity: {', '.join(color_identity)}
    Mana Cost: {mana_cost}
    Power/Toughness: {power}/{toughness}
    Keywords: {', '.join(keywords)}
    Legendary: {Legendary}
    Rules Text:
    {final_rules}
    """

        return {
            "card_display": card_display,
            "name": card_name,
            "rules_text": final_rules
        }

