import pandas as pd
import numpy as np
import requests
import re
import os
import json
from typing import Dict, List, Tuple, Optional

class DatasetCollector:
    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.hate_dict_path = os.path.join(self.base_dir, 'hate_speech_dictionary.json')

    def fetch_hate_speech_dictionary(self) -> Dict[str, float]:
        """
        Fetches and combines hate speech dictionaries from multiple sources.
        Returns a dictionary of terms with their severity scores.
        """
        try:
            # Fetch from multiple sources and combine
            hate_dict = {}
            
            # Source 1: HateBase API-like structured data
            hate_dict.update(self._fetch_hatebase_terms())
            
            # Source 2: Derogatory Terms Database
            hate_dict.update(self._fetch_derogatory_terms())
            
            # Source 3: Harmful Phrases Collection
            hate_dict.update(self._fetch_harmful_phrases())
            
            # Save the combined dictionary
            self._save_hate_dictionary(hate_dict)
            
            return hate_dict
            
        except Exception as e:
            print(f"Error fetching hate speech dictionary: {str(e)}")
            # Return default dictionary if fetch fails
            return self._get_default_hate_dict()

    def _fetch_hatebase_terms(self) -> Dict[str, float]:
        """
        Fetches terms from a HateBase-like structured format
        """
        # URL removed for ethical reasons - would need proper API access
        terms = {
            # Discriminatory terms (high severity)
            'bigot': 0.9, 'supremacist': 0.95, 'xenophobe': 0.9,
            'misogynist': 0.9, 'chauvinist': 0.85, 'racist': 0.9,
            
            # Dehumanizing terms (high severity)
            'subhuman': 1.0, 'vermin': 0.95, 'parasite': 0.9,
            'degenerate': 0.9, 'savage': 0.85, 'primitive': 0.8,
            
            # Gendered slurs (high severity)
            'feminazi': 0.9, 'thot': 0.85, 'incel': 0.8,
            'simp': 0.7, 'cuck': 0.8, 'white knight': 0.7,
            
            # Appearance-based (medium severity)
            'ugly': 0.6, 'fat': 0.6, 'disgusting': 0.7,
            'gross': 0.6, 'hideous': 0.7, 'deformed': 0.8,
            
            # Intelligence-based (medium severity)
            'moron': 0.7, 'imbecile': 0.7, 'retard': 0.9,
            'dimwit': 0.6, 'numbskull': 0.6, 'dunce': 0.6,
            
            # Character attacks (medium-high severity)
            'scumbag': 0.8, 'lowlife': 0.8, 'degenerate': 0.8,
            'sleazeball': 0.7, 'scoundrel': 0.7, 'miscreant': 0.7,
        }
        return terms

    def _fetch_derogatory_terms(self) -> Dict[str, float]:
        """
        Fetches modern derogatory terms and phrases
        """
        terms = {
            # Modern slang (medium-high severity)
            'basic': 0.6, 'karen': 0.7, 'boomer': 0.6,
            'snowflake': 0.7, 'triggered': 0.6, 'sjw': 0.7,
            
            # Social status (medium severity)
            'trailer trash': 0.8, 'ghetto': 0.8, 'ratchet': 0.7,
            'hood rat': 0.9, 'white trash': 0.8, 'peasant': 0.7,
            
            # Lifestyle attacks (medium severity)
            'gold digger': 0.8, 'attention seeker': 0.6,
            'clout chaser': 0.6, 'try hard': 0.5, 'poser': 0.5,
            
            # Relationship-based (medium-high severity)
            'friend zone': 0.6, 'beta': 0.7, 'alpha': 0.6,
            'chad': 0.6, 'stacy': 0.6, 'becky': 0.6,
        }
        return terms

    def _fetch_harmful_phrases(self) -> Dict[str, float]:
        """
        Fetches harmful phrases and expressions
        """
        phrases = {
            # Threats and wishes of harm
            'hope you die': 1.0, 'kill yourself': 1.0,
            'end your life': 1.0, 'should be dead': 1.0,
            'deserve to die': 1.0, 'better off dead': 1.0,
            
            # Dehumanizing phrases
            'waste of space': 0.9, 'waste of life': 0.9,
            'waste of oxygen': 0.9, 'waste of skin': 0.9,
            'worthless piece of': 0.9, 'absolute garbage': 0.9,
            
            # Discriminatory phrases
            'go back to': 0.9, 'your kind': 0.8,
            'you people': 0.7, 'like the rest': 0.7,
            'typical': 0.6, 'one of those': 0.6,
            
            # Sexual/Gender-based
            'belong to the streets': 0.95, 'belongs in the streets': 0.95,
            'for the streets': 0.9, 'nothing but a': 0.8,
            'asking for it': 0.9, 'was asking for': 0.9,
            
            # Dismissive/Gaslighting
            'cry about it': 0.7, 'cry more': 0.7,
            'stay mad': 0.7, 'cope harder': 0.7,
            'seethe more': 0.7, 'touch grass': 0.6,
            
            # Exclusionary
            'not welcome here': 0.8, 'dont belong here': 0.8,
            'get out of': 0.8, 'leave this': 0.7,
            'stick to': 0.6, 'stay in your': 0.7,
        }
        return phrases

    def _save_hate_dictionary(self, hate_dict: Dict[str, float]):
        """Saves the hate speech dictionary to a JSON file"""
        try:
            with open(self.hate_dict_path, 'w') as f:
                json.dump(hate_dict, f, indent=4)
        except Exception as e:
            print(f"Error saving hate dictionary: {str(e)}")

    def _get_default_hate_dict(self) -> Dict[str, float]:
        """Returns the default hate dictionary if saved version exists"""
        try:
            if os.path.exists(self.hate_dict_path):
                with open(self.hate_dict_path, 'r') as f:
                    return json.load(f)
        except Exception:
            pass
        return {}  # Return empty dict if all else fails

    def fetch_training_data(self) -> Optional[pd.DataFrame]:
        """
        Fetches training data from multiple sources
        Returns a DataFrame with 'text' and 'label' columns
        """
        try:
            # Combine datasets from multiple sources
            datasets = []
            
            # Twitter Hate Speech Dataset
            url = "https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv"
            df1 = pd.read_csv(url)
            df1 = df1[['tweet', 'class']].rename(columns={'tweet': 'text', 'class': 'label'})
            datasets.append(df1)
            
            # Wikipedia Toxic Comments Dataset (sample)
            url = "https://raw.githubusercontent.com/conversationai/unhealthy-conversations/main/unhealthy_full.csv"
            df2 = pd.read_csv(url, nrows=10000)  # Sample first 10k rows
            df2['label'] = df2['toxicity'].apply(lambda x: 1 if x > 0.5 else 0)
            df2 = df2[['text', 'label']]
            datasets.append(df2)
            
            # Combine all datasets
            final_df = pd.concat(datasets, ignore_index=True)
            
            # Clean the dataset
            final_df = final_df.dropna()
            final_df = final_df.drop_duplicates()
            
            return final_df
            
        except Exception as e:
            print(f"Error fetching training data: {str(e)}")
            return None

def download_hate_speech_datasets():
    """
    Download multiple hate speech datasets from various sources and save them in the data directory
    """
    # Ensure data directory exists
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    datasets = [
        {
            'name': 'Kaggle Hate Speech Dataset',
            'url': 'https://raw.githubusercontent.com/t-davidson/hate-speech-and-offensive-language/master/data/labeled_data.csv',
            'label_column': 'class',
            'text_column': 'tweet'
        },
        {
            'name': 'Toxic Comment Classification Challenge',
            'url': 'https://raw.githubusercontent.com/AshwanthRamji/Toxic-Comment-Classification/master/train.csv',
            'label_columns': ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        }
    ]
    
    all_datasets = []
    
    for dataset in datasets:
        try:
            # Download the dataset
            response = requests.get(dataset['url'])
            response.raise_for_status()
            
            # Save raw dataset in data directory
            filename = os.path.join(data_dir, f"{dataset['name'].replace(' ', '_')}.csv")
            with open(filename, 'wb') as f:
                f.write(response.content)
            
            # Read the dataset
            df = pd.read_csv(filename)
            
            # Process Kaggle Hate Speech Dataset
            if dataset['name'] == 'Kaggle Hate Speech Dataset':
                # 0: hate speech, 1: offensive language, 2: neither
                df['label'] = (df['class'] == 0).astype(int)
                processed_df = df[['tweet', 'label']]
                processed_df.columns = ['text', 'label']
            
            # Process Toxic Comment Dataset
            elif dataset['name'] == 'Toxic Comment Classification Challenge':
                # Combine toxic columns
                df['label'] = (df[dataset['label_columns']] > 0).any(axis=1).astype(int)
                processed_df = df[['comment_text', 'label']]
                processed_df.columns = ['text', 'label']
            
            all_datasets.append(processed_df)
            print(f"Successfully processed {dataset['name']}")
            
        except Exception as e:
            print(f"Error processing {dataset['name']}: {e}")
    
    if all_datasets:
        # Combine all datasets
        combined_df = pd.concat(all_datasets, ignore_index=True)
        
        # Save processed dataset
        processed_file = os.path.join(data_dir, 'processed_dataset.csv')
        combined_df.to_csv(processed_file, index=False)
        print(f"Combined dataset saved to {processed_file}")
        
        return combined_df
    
    return None

def clean_text(text):
    """
    Clean and preprocess text data
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove user mentions
    text = re.sub(r'@\w+', '', text)
    
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    
    # Remove numbers and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def update_hate_speech_detector(df):
    """
    Update the hate speech detector with new training data
    """
    import sys
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    
    from hate_speech_detector import HateSpeechDetector
    
    # Initialize detector with new data
    detector = HateSpeechDetector(df)
    detector.train_model()
    
    print("Hate speech detector updated with new training data")
    return detector

def main():
    print("Downloading and processing hate speech datasets...")
    combined_df = download_hate_speech_datasets()
    
    if combined_df is not None:
        print("\nDataset Summary:")
        print(f"Total samples: {len(combined_df)}")
        print(f"Hate speech samples: {combined_df['label'].sum()}")
        print(f"Non-hate speech samples: {len(combined_df) - combined_df['label'].sum()}")
        
        # Update the detector with new data
        detector = update_hate_speech_detector(combined_df)
    else:
        print("Failed to download and process datasets")

if __name__ == "__main__":
    collector = DatasetCollector()
    hate_dict = collector.fetch_hate_speech_dictionary()
    print(f"Collected {len(hate_dict)} hate speech terms and phrases")
    main()
