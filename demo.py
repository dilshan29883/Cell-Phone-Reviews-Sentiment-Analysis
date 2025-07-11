#!/usr/bin/env python3
"""
Quick Demo Script for Cell Phone Reviews Sentiment Analysis
Demonstrates the complete pipeline with sample data
"""

import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
warnings.filterwarnings('ignore')

print("🚀 Cell Phone Reviews Sentiment Analysis - Quick Demo")
print("=" * 60)

def create_sample_data():
    """Create sample data if the main dataset is not available"""
    print(" Creating sample data for demonstration...")
    
    sample_reviews = [
        {"reviewText": "This phone case is absolutely amazing! Great quality and fast shipping. Highly recommend!", "overall": 5.0},
        {"reviewText": "Terrible product. Broke after one day and customer service is awful. Don't waste your money.", "overall": 1.0},
        {"reviewText": "It's okay. Nothing special but does the job. Price is reasonable.", "overall": 3.0},
        {"reviewText": "Love this screen protector! Easy to install and crystal clear. Perfect fit for my phone.", "overall": 5.0},
        {"reviewText": "Cheaply made. The material feels flimsy and doesn't protect the phone well.", "overall": 2.0},
        {"reviewText": "Decent quality for the price. Not the best but works fine for everyday use.", "overall": 3.0},
        {"reviewText": "Excellent charger! Charges my phone super fast and the cable is durable.", "overall": 5.0},
        {"reviewText": "Complete waste of money. Stopped working after a week. Very disappointed.", "overall": 1.0},
        {"reviewText": "Average product. Does what it's supposed to do but nothing extraordinary.", "overall": 3.0},
        {"reviewText": "Outstanding quality! Exceeded my expectations. Will definitely buy again.", "overall": 5.0},
        {"reviewText": "Poor quality control. The product arrived damaged and the replacement was also defective.", "overall": 1.0},
        {"reviewText": "Good value for money. Not premium quality but works well for basic needs.", "overall": 4.0},
        {"reviewText": "Fast delivery and good packaging. The product works as described.", "overall": 4.0},
        {"reviewText": "Not impressed. The photos online made it look much better than reality.", "overall": 2.0},
        {"reviewText": "Perfect! Exactly what I was looking for. Great design and functionality.", "overall": 5.0},
        {"reviewText": "Mediocre product. There are better alternatives available at similar price.", "overall": 3.0},
        {"reviewText": "Highly recommend! Great customer service and excellent product quality.", "overall": 5.0},
        {"reviewText": "Don't buy this! It's a scam. Product doesn't match description at all.", "overall": 1.0},
        {"reviewText": "It's fine. Does the job but could be better designed.", "overall": 3.0},
        {"reviewText": "Amazing product! Best purchase I've made in a long time.", "overall": 5.0}
    ]
    
    return pd.DataFrame(sample_reviews)

def create_sentiment_labels(rating):
    """Convert ratings to sentiment labels"""
    if rating >= 4:
        return 'positive'
    elif rating <= 2:
        return 'negative'
    else:
        return 'neutral'

def clean_text(text):
    """Clean and preprocess text"""
    if pd.isna(text):
        return ""
    
    # Initialize lemmatizer and stop words
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove URLs, emails, HTML tags
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stop words and short words
    tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
    
    # Lemmatize
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    return ' '.join(tokens)

def train_demo_model():
    """Train a demo sentiment analysis model"""
    
    # Try to load the main dataset, fallback to sample data
    try:
        print("Attempting to load main dataset...")
        reviews = []
        with open('Cell_Phones_and_Accessories_5.json', 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10000:  # Use only first 10k reviews for demo
                    break
                try:
                    review = json.loads(line.strip())
                    if 'reviewText' in review and 'overall' in review:
                        reviews.append(review)
                except json.JSONDecodeError:
                    continue
        
        if len(reviews) > 100:
            df = pd.DataFrame(reviews)
            print(f"Loaded {len(df):,} reviews from main dataset")
        else:
            raise FileNotFoundError("Insufficient data in main dataset")
            
    except FileNotFoundError:
        print("Main dataset not found, using sample data...")
        df = create_sample_data()
        print(f"Created {len(df)} sample reviews")
    
    # Create sentiment labels
    df['sentiment_label'] = df['overall'].apply(create_sentiment_labels)
    
    print("\nDataset Overview:")
    print(f"Total Reviews: {len(df)}")
    print("Sentiment Distribution:")
    sentiment_counts = df['sentiment_label'].value_counts()
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment}: {count}")
    
    # Clean text
    print("\nCleaning text data...")
    df['clean_text'] = df['reviewText'].apply(clean_text)
    
    # Remove empty reviews
    df = df[df['clean_text'].str.len() > 0]
    print(f"Final dataset size: {len(df)} reviews")
    
    # Prepare features and labels
    X = df['clean_text']
    y = df['sentiment_label']
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"\nData Split:")
    print(f"Training set: {len(X_train)}")
    print(f"Test set: {len(X_test)}")
    
    # Create TF-IDF vectorizer and model pipeline
    print("\nTraining model...")
    
    # Create pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
        ('classifier', LogisticRegression(random_state=42))
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model trained successfully!")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
    
    # Save model components
    print("\nSaving model...")
    
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Save pipeline and encoder
    joblib.dump(pipeline, 'models/prediction_pipeline.pkl')
    joblib.dump(label_encoder, 'models/label_encoder.pkl')
    
    print("Model saved to models/ directory")
    
    return pipeline, label_encoder

def test_predictions(pipeline, label_encoder):
    """Test the trained model with sample predictions"""
    print("\nTesting Predictions:")
    print("-" * 40)
    
    test_texts = [
        "This phone case is amazing! Great quality and fast shipping.",
        "Terrible product. Broke after one day. Don't waste your money.",
        "It's okay. Nothing special but does the job.",
        "Love the design and functionality. Highly recommend!",
        "Poor quality and expensive. Not worth the money."
    ]
    
    for i, text in enumerate(test_texts, 1):
        # Clean text
        clean_text_sample = clean_text(text)
        
        # Make prediction
        prediction_encoded = pipeline.predict([clean_text_sample])[0]
        prediction_proba = pipeline.predict_proba([clean_text_sample])[0]
        
        # Decode prediction
        prediction = label_encoder.inverse_transform([prediction_encoded])[0]
        confidence = max(prediction_proba)
        
        print(f"\n{i}. Text: '{text[:60]}...'")
        print(f"   Sentiment: {prediction.upper()} (confidence: {confidence:.3f})")

def main():
    """Main demo function"""
    try:
        # Train model
        pipeline, label_encoder = train_demo_model()
        
        # Test predictions
        test_predictions(pipeline, label_encoder)
        
        print(f"\nDemo completed successfully!")
        print(f"Model files saved in 'models/' directory")
        print(f"You can now run the API: python api/app.py")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
