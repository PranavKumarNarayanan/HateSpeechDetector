import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import re
import os

class HateSpeechDetector:
    def __init__(self, dataset_path=None):
        # Default hardcoded training data
        self.training_data = {
            'text': [
                # Explicit Violent Threats
                'kill her',
                'she doesnt deserve a place in this world',
                'I want to kill you',
                'die already',
                'you should not exist',
                'I hope you get hurt',
                'someone should end your life',
                
                # Dehumanizing Language
                'you are less than human',
                'worthless piece of trash',
                'you dont deserve to live',
                'world would be better without you',
                'you are a waste of space',
                
                # Gender-based Hate
                'women are useless',
                'she is only good for one thing',
                'females are inferior',
                'women should know their place',
                
                # Racial and Discriminatory Comments
                'your kind is a problem',
                'go back to your country',
                'you dont belong here',
                'all of your people are the same',
                
                # Explicit Hate Speech
                'I hate you',
                'you are garbage',
                'nobody wants you here',
                'you are disgusting',
                
                # Non-Hate Speech Examples - General Positive
                'Great weather today!',
                'I love this community',
                'Everyone deserves respect',
                'Let\'s work together',
                'You are awesome',
                'What a beautiful day',
                'Welcome everyone',
                'I appreciate your help',
                'Thank you for your kindness',
                'This is a wonderful place',
                
                # Non-Hate Speech - Compliments
                'I love your outfit',
                'Your hair looks amazing',
                'That dress suits you perfectly',
                'You have a beautiful smile',
                'Your style is fantastic',
                'You look great today',
                'Nice choice of colors',
                'That looks good on you',
                'You have great taste',
                'Love what you did there',
                
                # Non-Hate Speech - Supportive Comments
                'Keep up the good work',
                'You can do this',
                'I believe in you',
                'You\'re making progress',
                'That\'s a great idea',
                'Well done',
                'Proud of you',
                'You\'re doing great',
                'Thanks for sharing',
                'This is inspiring'
            ],
            'label': [
                # Hate Speech Labels
                1, 1, 1, 1, 1, 1, 1,  # Violent Threats
                1, 1, 1, 1, 1,  # Dehumanizing Language
                1, 1, 1, 1,  # Gender-based Hate
                1, 1, 1, 1,  # Racial Comments
                1, 1, 1, 1,  # Explicit Hate
                
                # Non-Hate Speech Labels - General Positive
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                
                # Non-Hate Speech - Compliments
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                
                # Non-Hate Speech - Supportive Comments
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            ]
        }
        
        # Load additional dataset if provided
        if dataset_path and os.path.exists(dataset_path):
            try:
                df = pd.read_csv(dataset_path)
                
                # Ensure the dataframe has 'text' and 'label' columns
                if 'text' in df.columns and 'label' in df.columns:
                    # Extend training data with new dataset
                    self.training_data['text'].extend(df['text'].tolist())
                    self.training_data['label'].extend(df['label'].tolist())
                    print(f"Loaded additional {len(df)} samples from {dataset_path}")
                else:
                    print(f"Warning: Dataset {dataset_path} does not have required columns")
            except Exception as e:
                print(f"Error loading dataset: {e}")
        
        # Create pipeline with adjusted parameters
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                stop_words='english',
                ngram_range=(1, 2),
                max_features=5000,
                min_df=2  # Minimum document frequency
            )),
            ('classifier', MultinomialNB(alpha=0.5))  # Increased smoothing
        ])
    
    def clean_text(self, text):
        """
        Advanced text cleaning method
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers but keep apostrophes
        text = re.sub(r'[^a-zA-Z\s\']', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def train_model(self):
        """
        Train the hate speech detection model
        """
        # Clean the text data
        cleaned_texts = [self.clean_text(text) for text in self.training_data['text']]
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            cleaned_texts, 
            self.training_data['label'], 
            test_size=0.2, 
            random_state=42,
            stratify=self.training_data['label']  # Ensure balanced split
        )
        
        # Fit the pipeline
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate the model
        accuracy = self.pipeline.score(X_test, y_test)
        print(f"Model validation accuracy: {accuracy * 100:.2f}%")
        
        return self
    
    def predict(self, text):
        """
        Predict if text contains hate speech
        Returns probability of hate speech
        """
        # Clean the input text
        cleaned_text = self.clean_text(text)
        
        # Predict probabilities
        proba = self.pipeline.predict_proba([cleaned_text])[0]
        
        # Return probability of hate speech (second class)
        return {
            'hate_speech_probability': proba[1],
            'is_hate_speech': proba[1] > 0.6  # Increased threshold
        }

def main():
    # Path to the combined dataset
    dataset_path = os.path.join(os.path.dirname(__file__), 'combined_hate_speech_dataset.csv')
    
    # Initialize and train the detector with the dataset
    detector = HateSpeechDetector(dataset_path).train_model()
    
    print("\nHate Speech Detection Interactive Tool")
    print("-------------------------------------")
    print("Enter comments to analyze. Type 'quit' to exit.")
    
    while True:
        try:
            # Get user input
            text = input("\nEnter a comment to analyze (or 'quit' to exit): ").strip()
            
            # Check for exit condition
            if text.lower() == 'quit':
                print("Exiting hate speech detection tool. Goodbye!")
                break
            
            # Skip empty input
            if not text:
                print("Please enter some text to analyze.")
                continue
            
            # Predict hate speech
            result = detector.predict(text)
            
            # Display results
            print("\n--- Analysis Results ---")
            print(f"Comment: '{text}'")
            print(f"Hate Speech Probability: {result['hate_speech_probability']:.2f}")
            print(f"Is Hate Speech: {result['is_hate_speech']}")
            
            # Provide additional context based on probability
            if result['is_hate_speech']:
                if result['hate_speech_probability'] > 0.9:
                    print("WARNING: This comment contains extremely harmful language.")
                elif result['hate_speech_probability'] > 0.7:
                    print("CAUTION: This comment contains potentially offensive content.")
            
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
