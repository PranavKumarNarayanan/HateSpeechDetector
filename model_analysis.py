import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from hate_speech_detector import HateSpeechDetector

def load_and_prepare_data():
    """Load and prepare the dataset for analysis."""
    dataset = pd.read_csv('datasets/processed_dataset.csv')
    
    # Split the data
    X = dataset['text']
    y = dataset['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize and fit detector
    detector = HateSpeechDetector()
    
    # Get predictions
    y_pred = []
    y_pred_proba = []
    
    for text in X_test:
        is_hate_speech, probability, details = detector.detect_hate_speech(text)
        y_pred.append(1 if is_hate_speech else 0)
        y_pred_proba.append(probability)
    
    return y_test, np.array(y_pred), np.array(y_pred_proba), detector

def plot_confusion_matrix(y_test, y_pred):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('visualizations/confusion_matrix.png')
    plt.close()

def plot_roc_curve(y_test, y_pred_proba):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('visualizations/roc_curve.png')
    plt.close()

def plot_feature_importance(detector):
    """Plot top features importance."""
    # Get feature names
    feature_names = detector.vectorizer.get_feature_names_out()
    
    # Prepare dataset for feature importance calculation
    dataset = pd.read_csv('datasets/processed_dataset.csv')
    
    # Remove NaN values
    dataset = dataset.dropna(subset=['text'])
    
    # Sample a subset of data to calculate feature importance
    sample_size = min(1000, len(dataset))
    dataset_sample = dataset.sample(n=sample_size, random_state=42)
    
    # Vectorize the text
    X = detector.vectorizer.transform(dataset_sample['text'])
    
    # Use the trained model to get feature importances
    # We'll use the predict_proba method to get feature contributions
    feature_importances = np.zeros(len(feature_names))
    
    for idx in range(X.shape[0]):
        text_vec = X[idx]
        # Get probability of hate speech
        prob = detector.model.predict_proba(text_vec.reshape(1, -1))[0][1]
        
        # Calculate feature contributions
        feature_contributions = text_vec.toarray()[0] * prob
        feature_importances += np.abs(feature_contributions)
    
    # Normalize feature importances
    feature_importances /= sample_size
    
    # Get top 20 features
    top_n = 20
    top_indices = feature_importances.argsort()[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    top_importances = feature_importances[top_indices]
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.barh(top_features, top_importances)
    plt.title(f'Top {top_n} Most Important Features for Hate Speech Detection')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.savefig('visualizations/feature_importance.png')
    plt.close()

def plot_class_distribution(y_test, y_pred):
    """Plot actual vs predicted class distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot actual distribution
    sns.countplot(x=y_test, ax=ax1)
    ax1.set_title('Actual Class Distribution')
    ax1.set_xlabel('Class')
    ax1.set_xticklabels(['Non-Hate Speech', 'Hate Speech'])
    
    # Plot predicted distribution
    sns.countplot(x=y_pred, ax=ax2)
    ax2.set_title('Predicted Class Distribution')
    ax2.set_xlabel('Class')
    ax2.set_xticklabels(['Non-Hate Speech', 'Hate Speech'])
    
    plt.tight_layout()
    plt.savefig('visualizations/class_distribution.png')
    plt.close()

def save_classification_report(y_test, y_pred):
    """Save classification report as text file."""
    report = classification_report(y_test, y_pred, 
                                 target_names=['Non-Hate Speech', 'Hate Speech'])
    with open('visualizations/classification_report.txt', 'w') as f:
        f.write("Hate Speech Detection Model Performance Report\n")
        f.write("===========================================\n\n")
        f.write(report)

def main():
    # Create visualizations directory if it doesn't exist
    import os
    os.makedirs('visualizations', exist_ok=True)
    
    # Load and prepare data
    print("Loading and preparing data...")
    y_test, y_pred, y_pred_proba, detector = load_and_prepare_data()
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_proba)
    plot_feature_importance(detector)
    plot_class_distribution(y_test, y_pred)
    save_classification_report(y_test, y_pred)
    
    print("\nVisualization complete! Check the 'visualizations' directory for the following files:")
    print("1. confusion_matrix.png - Shows model's prediction accuracy")
    print("2. roc_curve.png - Shows model's discrimination ability")
    print("3. feature_importance.png - Shows top important features")
    print("4. class_distribution.png - Shows class distribution comparison")
    print("5. classification_report.txt - Detailed classification metrics")

if __name__ == "__main__":
    main()
