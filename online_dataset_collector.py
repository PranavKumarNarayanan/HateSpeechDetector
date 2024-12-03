import pandas as pd
import numpy as np
import requests
import re

def download_hate_speech_datasets():
    """
    Download multiple hate speech datasets from various sources
    """
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
            
            # Save raw dataset
            filename = f"/home/pranavks/Documents/Python Project/{dataset['name'].replace(' ', '_')}.csv"
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
            
            # Save processed dataset
            processed_filename = f"/home/pranavks/Documents/Python Project/{dataset['name'].replace(' ', '_')}_processed.csv"
            processed_df.to_csv(processed_filename, index=False)
            
            all_datasets.append(processed_df)
            
            print(f"Successfully downloaded and processed {dataset['name']}")
            print(f"Total samples: {len(processed_df)}")
            print(f"Hate speech samples: {processed_df['label'].sum()}")
            print("---")
        
        except Exception as e:
            print(f"Error processing {dataset.get('name', 'Unknown dataset')}: {e}")
    
    # Combine all datasets
    if all_datasets:
        combined_df = pd.concat(all_datasets, ignore_index=True)
        combined_filename = "/home/pranavks/Documents/Python Project/combined_hate_speech_dataset.csv"
        combined_df.to_csv(combined_filename, index=False)
        
        print("\nCombined Dataset Summary:")
        print(f"Total samples: {len(combined_df)}")
        print(f"Hate speech samples: {combined_df['label'].sum()}")
        print(f"Saved to: {combined_filename}")
        
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
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespaces
    text = ' '.join(text.split())
    
    return text

def update_hate_speech_detector(df):
    """
    Update the hate speech detector with new training data
    """
    import sys
    sys.path.append('/home/pranavks/Documents/Python Project')
    
    from hate_speech_detector import HateSpeechDetector
    
    # Clean the text
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    # Initialize detector
    detector = HateSpeechDetector()
    
    # Extend training data
    detector.training_data['text'].extend(df['cleaned_text'].tolist())
    detector.training_data['label'].extend(df['label'].tolist())
    
    # Retrain the model
    detector.train_model()
    
    print("\nModel Updated Successfully!")
    print(f"Total training samples: {len(detector.training_data['text'])}")

def main():
    # Download and process datasets
    combined_df = download_hate_speech_datasets()
    
    if combined_df is not None:
        # Update hate speech detector
        update_hate_speech_detector(combined_df)

if __name__ == "__main__":
    main()
