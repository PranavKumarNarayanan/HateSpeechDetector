import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
import re
import os
from typing import Dict, List, Tuple, Optional

class HateSpeechDetector:
    def __init__(self):
        """Initialize the hate speech detector with Twitter dataset"""
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.dataset_path = os.path.join(self.base_dir, 'datasets/processed_dataset.csv')
        
        # Initialize vectorizer with enhanced parameters
        self.vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 5),
            min_df=2,
            max_df=0.95
        )
        
        # Initialize model with calibration for better probability estimates
        self.model = CalibratedClassifierCV(
            LogisticRegression(
                class_weight='balanced',
                random_state=42,
                max_iter=1000
            )
        )
        
        self.threshold = 0.35
        self.hate_dict = self._load_hate_dictionary()
        
        # Load and train on the Twitter dataset
        self.dataset = self._load_dataset()
        if self.dataset is not None:
            self.train_model()
        else:
            raise ValueError("No dataset available for training")

    def _load_dataset(self) -> pd.DataFrame:
        """Load the processed Twitter dataset"""
        try:
            if os.path.exists(self.dataset_path):
                df = pd.read_csv(self.dataset_path)
                print(f"Loaded {len(df)} examples from Twitter dataset")
                return df
            else:
                print("Dataset not found at:", self.dataset_path)
                return self._get_default_dataset()
        except Exception as e:
            print(f"Error loading dataset: {str(e)}")
            return self._get_default_dataset()

    def _get_default_dataset(self) -> pd.DataFrame:
        """Fallback dataset if Twitter data is unavailable"""
        hate_examples = [
            "I hate all of them, they should die",
            "You're worthless and deserve to suffer",
            "Go kill yourself",
            "All of them are subhuman",
            "They're nothing but parasites"
        ]
        
        non_hate_examples = [
            "What a beautiful day!",
            "I love this movie so much",
            "Great job on the presentation",
            "Looking forward to seeing you",
            "This is really interesting"
        ]
        
        texts = hate_examples + non_hate_examples
        labels = [1] * len(hate_examples) + [0] * len(non_hate_examples)
        
        return pd.DataFrame({
            'text': texts,
            'label': labels
        })

    def _load_hate_dictionary(self) -> Dict[str, float]:
        """Load the hate speech dictionary"""
        from online_dataset_collector import DatasetCollector
        collector = DatasetCollector()
        return collector.fetch_hate_speech_dictionary()

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove user mentions
        text = re.sub(r'@\w+', '', text)
        
        # Remove hashtags
        text = re.sub(r'#\w+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.strip()

    def train_model(self):
        """Train the hate speech detection model"""
        # Clean the text data
        self.dataset['cleaned_text'] = self.dataset['text'].apply(self.clean_text)
        
        # Split into training and validation sets
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            self.dataset['cleaned_text'],
            self.dataset['label'],
            test_size=0.2,
            random_state=42,
            stratify=self.dataset['label']
        )
        
        print("\nTraining model on Twitter dataset...")
        print(f"Training examples: {len(X_train)}")
        print(f"Validation examples: {len(X_val)}")
        
        # Fit the vectorizer and transform the text
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_val_vec = self.vectorizer.transform(X_val)
        
        # Train the model
        print("Training classifier...")
        self.model.fit(X_train_vec, y_train)
        
        # Calculate and print validation accuracy
        val_accuracy = self.model.score(X_val_vec, y_val)
        print(f"Validation accuracy: {val_accuracy:.4f}")

    def _calculate_rule_based_score(self, text: str) -> float:
        """Calculate rule-based hate speech score"""
        text = text.lower()
        score = 0.0
        total_weight = 0.0
        
        # Check for hate speech terms and phrases
        for term, severity in self.hate_dict.items():
            if term.lower() in text:
                score += severity
                total_weight += 1.0
        
        # Return normalized score
        return score / max(total_weight, 1.0)

    def detect_hate_speech(self, text: str) -> Tuple[bool, float, Dict[str, float]]:
        """
        Detect hate speech in text using both ML and rule-based approaches
        Returns: (is_hate_speech, probability, details)
        """
        # Clean the text
        cleaned_text = self.clean_text(text)
        
        # Get ML probability
        text_vec = self.vectorizer.transform([cleaned_text])
        ml_prob = self.model.predict_proba(text_vec)[0][1]
        
        # Get rule-based score
        rule_score = self._calculate_rule_based_score(cleaned_text)
        
        # Combine probabilities (50% ML, 50% rule-based)
        final_prob = 0.5 * ml_prob + 0.5 * rule_score
        
        # Classify as hate speech if probability exceeds threshold
        is_hate_speech = final_prob >= self.threshold
        
        # Return detailed results
        details = {
            'ml_probability': ml_prob,
            'rule_based_score': rule_score,
            'final_probability': final_prob,
            'threshold': self.threshold
        }
        
        return is_hate_speech, final_prob, details

def main():
    # Initialize detector with Twitter dataset
    print("Initializing Hate Speech Detector with Twitter dataset...")
    detector = HateSpeechDetector()
    
    print("\nHate Speech Detector initialized. Enter text to analyze (or 'q' to quit):")
    
    while True:
        try:
            # Get input from user
            text = input("\nEnter text: ").strip()
            
            # Check for quit command
            if text.lower() == 'q':
                break
            
            # Skip empty input
            if not text:
                continue
            
            # Analyze text
            is_hate, prob, details = detector.detect_hate_speech(text)
            
            # Print results
            print("\nAnalysis Results:")
            print(f"Text: {text}")
            print(f"Is Hate Speech: {is_hate}")
            print(f"Confidence: {prob:.2%}")
            print("\nDetailed Scores:")
            print(f"- ML Model Score: {details['ml_probability']:.2%}")
            print(f"- Rule-based Score: {details['rule_based_score']:.2%}")
            print(f"- Final Score: {details['final_probability']:.2%}")
            print(f"- Threshold: {details['threshold']:.2%}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {str(e)}")
            continue
    
    print("\nThank you for using the Hate Speech Detector!")

if __name__ == "__main__":
    main()
