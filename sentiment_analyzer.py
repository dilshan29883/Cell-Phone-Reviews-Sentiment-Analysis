#!/usr/bin/env python3
"""
Cell Phone Reviews Sentiment Analysis Pipeline
Comprehensive implementation of all sentiment analysis steps
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import re
import string
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# NLP Libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from textblob import TextBlob

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold

# Deep Learning (optional)
try:
    import torch
    import torch.nn as nn
    from transformers import BertTokenizer, BertForSequenceClassification
    DEEP_LEARNING_AVAILABLE = True
except ImportError:
    DEEP_LEARNING_AVAILABLE = False
    print("Deep learning libraries not available. Will use classical ML only.")

# Visualization
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from wordcloud import WordCloud

# Utilities
import joblib
import pickle
import os
from pathlib import Path

class SentimentAnalyzer:
    def __init__(self, data_path):
        """Initialize the sentiment analyzer with dataset path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.models = {}
        self.vectorizers = {}
        self.results = {}
        
        # Create directories
        Path("models").mkdir(exist_ok=True)
        Path("visualizations").mkdir(exist_ok=True)
        
        # Download NLTK data
        self._download_nltk_data()
        
    def _download_nltk_data(self):
        """Download required NLTK datasets"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            nltk.download('wordnet')
            
        try:
            nltk.data.find('corpora/omw-1.4')
        except LookupError:
            nltk.download('omw-1.4')
    
    def load_data(self):
        """C. Data Exploration & Understanding - Load and analyze dataset"""
        print("🔍 Loading and analyzing dataset...")
        
        # Load JSON data
        reviews = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    review = json.loads(line.strip())
                    reviews.append(review)
                except json.JSONDecodeError:
                    continue
        
        # Convert to DataFrame
        self.df = pd.DataFrame(reviews)
        
        # Basic info
        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total reviews: {len(self.df):,}")
        print(f"📋 Features: {list(self.df.columns)}")
        print(f"🔍 Shape: {self.df.shape}")
        
        # Check for missing values
        print("\n📋 Missing values:")
        print(self.df.isnull().sum())
        
        return self.df
    
    def analyze_data(self):
        """Analyze dataset characteristics"""
        print("\n🔍 Analyzing dataset characteristics...")
        
        # Create sentiment labels from ratings
        self.df['sentiment_label'] = self.df['overall'].apply(self._create_sentiment_labels)
        
        # Basic statistics
        print(f"📊 Rating distribution:")
        print(self.df['overall'].value_counts().sort_index())
        
        print(f"\n📊 Sentiment distribution:")
        sentiment_counts = self.df['sentiment_label'].value_counts()
        print(sentiment_counts)
        
        # Check for data balance
        total_reviews = len(self.df)
        balance_ratio = sentiment_counts.min() / sentiment_counts.max()
        print(f"\n⚖️  Dataset balance ratio: {balance_ratio:.2f}")
        if balance_ratio < 0.5:
            print("⚠️  Dataset is imbalanced!")
        else:
            print("✅ Dataset is reasonably balanced")
        
        # Text length analysis
        self.df['text_length'] = self.df['reviewText'].astype(str).apply(len)
        print(f"\n📏 Text length statistics:")
        print(self.df['text_length'].describe())
        
        # Remove rows with missing review text
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['reviewText'])
        final_count = len(self.df)
        print(f"\n🧹 Removed {initial_count - final_count} rows with missing review text")
        print(f"📊 Final dataset size: {final_count:,} reviews")
        
        # Create visualizations
        self._create_exploration_visualizations()
        
        return self.df
    
    def _create_sentiment_labels(self, rating):
        """Convert ratings to sentiment labels"""
        if rating >= 4:
            return 'positive'
        elif rating <= 2:
            return 'negative'
        else:
            return 'neutral'
    
    def _create_exploration_visualizations(self):
        """Create data exploration visualizations"""
        print("📊 Creating visualizations...")
        
        # Rating distribution
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Rating Distribution', 'Sentiment Distribution', 
                          'Text Length Distribution', 'Reviews Over Time'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Rating distribution
        rating_counts = self.df['overall'].value_counts().sort_index()
        fig.add_trace(
            go.Bar(x=rating_counts.index, y=rating_counts.values, name="Ratings"),
            row=1, col=1
        )
        
        # Sentiment distribution
        sentiment_counts = self.df['sentiment_label'].value_counts()
        fig.add_trace(
            go.Pie(labels=sentiment_counts.index, values=sentiment_counts.values, name="Sentiment"),
            row=1, col=2
        )
        
        # Text length distribution
        fig.add_trace(
            go.Histogram(x=self.df['text_length'], name="Text Length"),
            row=2, col=1
        )
        
        # Reviews over time (if timestamp available)
        if 'reviewTime' in self.df.columns:
            time_counts = self.df['reviewTime'].value_counts().head(20)
            fig.add_trace(
                go.Scatter(x=time_counts.index, y=time_counts.values, mode='lines+markers', name="Reviews Over Time"),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=False, title_text="Dataset Exploration Dashboard")
        fig.write_html("visualizations/data_exploration.html")
        print("✅ Visualizations saved to visualizations/data_exploration.html")
        
        # Word cloud for positive and negative reviews
        self._create_wordclouds()
    
    def _create_wordclouds(self):
        """Create word clouds for different sentiments"""
        print("☁️  Creating word clouds...")
        
        # Positive reviews word cloud
        positive_text = ' '.join(self.df[self.df['sentiment_label'] == 'positive']['reviewText'].astype(str))
        positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_text)
        
        # Negative reviews word cloud
        negative_text = ' '.join(self.df[self.df['sentiment_label'] == 'negative']['reviewText'].astype(str))
        negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_text)
        
        # Plot word clouds
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        ax1.imshow(positive_wordcloud, interpolation='bilinear')
        ax1.set_title('Positive Reviews - Word Cloud', fontsize=16)
        ax1.axis('off')
        
        ax2.imshow(negative_wordcloud, interpolation='bilinear')
        ax2.set_title('Negative Reviews - Word Cloud', fontsize=16)
        ax2.axis('off')
        
        plt.tight_layout()
        plt.savefig('visualizations/wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ Word clouds saved to visualizations/wordclouds.png")
    
    def preprocess_data(self):
        """D. Data Preprocessing - Clean and prepare text data"""
        print("\n🧹 Starting data preprocessing...")
        
        # Initialize preprocessing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        
        # Apply preprocessing
        print("📝 Cleaning text data...")
        self.df['clean_text'] = self.df['reviewText'].astype(str).apply(self._clean_text)
        
        # Remove rows with empty clean text
        initial_count = len(self.df)
        self.df = self.df[self.df['clean_text'].str.len() > 0]
        final_count = len(self.df)
        print(f"🧹 Removed {initial_count - final_count} rows with empty clean text")
        
        print("✅ Data preprocessing completed!")
        print(f"📊 Final dataset size: {final_count:,} reviews")
        
        # Show sample of cleaned data
        print("\n📋 Sample of cleaned data:")
        sample_df = self.df[['reviewText', 'clean_text', 'sentiment_label']].head(3)
        for idx, row in sample_df.iterrows():
            print(f"\nOriginal: {row['reviewText'][:100]}...")
            print(f"Cleaned:  {row['clean_text'][:100]}...")
            print(f"Sentiment: {row['sentiment_label']}")
        
        return self.df
    
    def _clean_text(self, text):
        """Clean individual text samples"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and short words
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        
        return ' '.join(tokens)
    
    def feature_extraction(self):
        """E. Feature Extraction - Convert text to numerical vectors"""
        print("\n🔢 Extracting features from text data...")
        
        # Prepare data
        X = self.df['clean_text']
        y = self.df['sentiment_label']
        
        # TF-IDF Vectorization
        print("📊 Creating TF-IDF features...")
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95
        )
        
        X_tfidf = self.tfidf_vectorizer.fit_transform(X)
        print(f"✅ TF-IDF feature matrix shape: {X_tfidf.shape}")
        
        # Store vectorizer for later use
        self.vectorizers['tfidf'] = self.tfidf_vectorizer
        
        # Save vectorizer
        joblib.dump(self.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
        
        return X_tfidf, y
    
    def split_data(self, X, y):
        """F. Split Dataset - Create train/test splits"""
        print("\n📊 Splitting dataset...")
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        print(f"✅ Dataset split completed!")
        print(f"📊 Training set size: {self.X_train.shape[0]:,}")
        print(f"📊 Test set size: {self.X_test.shape[0]:,}")
        print(f"📊 Feature dimensions: {self.X_train.shape[1]:,}")
        
        # Save label encoder
        joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_models(self):
        """G. Model Selection & Training - Train multiple models"""
        print("\n🤖 Training multiple models...")
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(random_state=42, probability=True),
            'Naive Bayes': MultinomialNB(),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100)
        }
        
        # Train each model
        for name, model in models.items():
            print(f"🏋️  Training {name}...")
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Store model and predictions
            self.models[name] = {
                'model': model,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            # Save model
            joblib.dump(model, f'models/{name.lower().replace(" ", "_")}_model.pkl')
            
            print(f"✅ {name} trained and saved!")
        
        return self.models
    
    def evaluate_models(self):
        """H. Model Evaluation - Evaluate all trained models"""
        print("\n📊 Evaluating model performance...")
        
        results = {}
        
        for name, model_data in self.models.items():
            print(f"\n🔍 Evaluating {name}...")
            
            model = model_data['model']
            y_pred = model_data['y_pred']
            y_pred_proba = model_data['y_pred_proba']
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, y_pred, average='weighted')
            
            # Store results
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': confusion_matrix(self.y_test, y_pred),
                'classification_report': classification_report(self.y_test, y_pred, 
                                                             target_names=self.label_encoder.classes_)
            }
            
            # Print results
            print(f"📊 Accuracy: {accuracy:.4f}")
            print(f"📊 Precision: {precision:.4f}")
            print(f"📊 Recall: {recall:.4f}")
            print(f"📊 F1-Score: {f1:.4f}")
            
            # Cross-validation
            cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=5, scoring='accuracy')
            results[name]['cv_accuracy'] = cv_scores.mean()
            results[name]['cv_std'] = cv_scores.std()
            
            print(f"📊 CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.results = results
        
        # Create comparison visualization
        self._create_model_comparison()
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['f1_score'])
        print(f"\n🏆 Best model: {best_model_name} (F1-Score: {results[best_model_name]['f1_score']:.4f})")
        
        return results
    
    def _create_model_comparison(self):
        """Create model comparison visualizations"""
        print("📊 Creating model comparison visualizations...")
        
        # Prepare data for visualization
        model_names = list(self.results.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        
        # Create comparison chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Accuracy', 'Precision', 'Recall', 'F1-Score')
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        for i, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in model_names]
            
            row = (i // 2) + 1
            col = (i % 2) + 1
            
            fig.add_trace(
                go.Bar(x=model_names, y=values, name=metric.title(), 
                       marker_color=colors[i], showlegend=False),
                row=row, col=col
            )
        
        fig.update_layout(height=600, title_text="Model Performance Comparison")
        fig.write_html("visualizations/model_comparison.html")
        
        # Create confusion matrices
        self._create_confusion_matrices()
        
        print("✅ Model comparison visualizations saved!")
    
    def _create_confusion_matrices(self):
        """Create confusion matrix visualizations"""
        n_models = len(self.models)
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, (name, results) in enumerate(self.results.items()):
            if i >= 4:  # Only show first 4 models
                break
                
            cm = results['confusion_matrix']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.label_encoder.classes_,
                       yticklabels=self.label_encoder.classes_,
                       ax=axes[i])
            axes[i].set_title(f'{name} - Confusion Matrix')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('visualizations/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def improve_models(self):
        """I. Model Improvement - Hyperparameter tuning"""
        print("\n🔧 Improving models with hyperparameter tuning...")
        
        # Hyperparameter tuning for best performing models
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        
        if best_model_name == 'Logistic Regression':
            print("🔧 Tuning Logistic Regression...")
            param_grid = {
                'C': [0.1, 1, 10],
                'solver': ['liblinear', 'lbfgs']
            }
            
            grid_search = GridSearchCV(
                LogisticRegression(random_state=42, max_iter=1000),
                param_grid, cv=5, scoring='f1_weighted', n_jobs=-1
            )
            
        elif best_model_name == 'Random Forest':
            print("🔧 Tuning Random Forest...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5]
            }
            
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid, cv=3, scoring='f1_weighted', n_jobs=-1
            )
        
        else:
            print(f"🔧 Tuning {best_model_name}...")
            # Use the original model for other cases
            grid_search = self.models[best_model_name]['model']
        
        if hasattr(grid_search, 'fit'):
            grid_search.fit(self.X_train, self.y_train)
            
            if hasattr(grid_search, 'best_params_'):
                print(f"✅ Best parameters: {grid_search.best_params_}")
                print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
                
                # Save improved model
                self.best_model = grid_search.best_estimator_
                joblib.dump(self.best_model, 'models/best_model.pkl')
                
                # Evaluate improved model
                y_pred_improved = self.best_model.predict(self.X_test)
                accuracy_improved = accuracy_score(self.y_test, y_pred_improved)
                
                print(f"🏆 Improved model accuracy: {accuracy_improved:.4f}")
                print(f"📈 Improvement: {accuracy_improved - self.results[best_model_name]['accuracy']:.4f}")
    
    def create_prediction_pipeline(self):
        """J. Deployment Preparation - Create prediction pipeline"""
        print("\n🔧 Creating prediction pipeline...")
        
        # Get best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        best_model = self.models[best_model_name]['model']
        
        # Create pipeline
        self.prediction_pipeline = Pipeline([
            ('vectorizer', self.tfidf_vectorizer),
            ('classifier', best_model)
        ])
        
        # Save complete pipeline
        joblib.dump(self.prediction_pipeline, 'models/prediction_pipeline.pkl')
        
        print("✅ Prediction pipeline created and saved!")
        
        # Test pipeline
        sample_text = "This phone case is amazing! Great quality and fast shipping."
        prediction = self.predict_sentiment(sample_text)
        print(f"🧪 Pipeline test - Text: '{sample_text}'")
        print(f"🧪 Predicted sentiment: {prediction}")
        
        return self.prediction_pipeline
    
    def predict_sentiment(self, text):
        """Predict sentiment for new text"""
        if not hasattr(self, 'prediction_pipeline'):
            # Load pipeline if not available
            try:
                self.prediction_pipeline = joblib.load('models/prediction_pipeline.pkl')
                self.label_encoder = joblib.load('models/label_encoder.pkl')
            except FileNotFoundError:
                print("❌ Pipeline not found. Please train models first.")
                return None
        
        # Initialize cleaning tools if not available
        if not hasattr(self, 'stop_words'):
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
        
        # Clean text
        clean_text = self._clean_text(text)
        
        # Make prediction
        prediction_encoded = self.prediction_pipeline.predict([clean_text])[0]
        prediction_proba = self.prediction_pipeline.predict_proba([clean_text])[0]
        
        # Decode prediction
        prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get confidence scores
        confidence_scores = {
            class_name: float(prob) 
            for class_name, prob in zip(self.label_encoder.classes_, prediction_proba)
        }
        
        return {
            'sentiment': prediction,
            'confidence': float(max(prediction_proba)),
            'probabilities': confidence_scores
        }
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        print("\n📊 Generating comprehensive report...")
        
        report = f"""
# Sentiment Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Dataset Summary
- Total Reviews: {len(self.df):,}
- Features: {list(self.df.columns)}
- Sentiment Distribution:
"""
        
        # Add sentiment distribution
        sentiment_counts = self.df['sentiment_label'].value_counts()
        for sentiment, count in sentiment_counts.items():
            percentage = (count / len(self.df)) * 100
            report += f"  - {sentiment.title()}: {count:,} ({percentage:.1f}%)\n"
        
        report += "\n## Model Performance\n"
        
        # Add model results
        for model_name, results in self.results.items():
            report += f"\n### {model_name}\n"
            report += f"- Accuracy: {results['accuracy']:.4f}\n"
            report += f"- Precision: {results['precision']:.4f}\n"
            report += f"- Recall: {results['recall']:.4f}\n"
            report += f"- F1-Score: {results['f1_score']:.4f}\n"
            report += f"- CV Accuracy: {results['cv_accuracy']:.4f} (+/- {results['cv_std']:.4f})\n"
        
        # Find best model
        best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['f1_score'])
        report += f"\n## Best Model: {best_model_name}\n"
        report += f"F1-Score: {self.results[best_model_name]['f1_score']:.4f}\n"
        
        # Save report
        with open('sentiment_analysis_report.md', 'w') as f:
            f.write(report)
        
        print("✅ Report saved as 'sentiment_analysis_report.md'")
        
        return report

def main():
    """Main execution function"""
    print("🚀 Starting Cell Phone Reviews Sentiment Analysis")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer("Cell_Phones_and_Accessories_5.json")
    
    try:
        # C. Data Exploration & Understanding
        df = analyzer.load_data()
        df = analyzer.analyze_data()
        
        # D. Data Preprocessing
        df = analyzer.preprocess_data()
        
        # E. Feature Extraction
        X, y = analyzer.feature_extraction()
        
        # F. Split Dataset
        X_train, X_test, y_train, y_test = analyzer.split_data(X, y)
        
        # G. Model Selection & Training
        models = analyzer.train_models()
        
        # H. Model Evaluation
        results = analyzer.evaluate_models()
        
        # I. Model Improvement
        analyzer.improve_models()
        
        # J. Deployment Preparation
        pipeline = analyzer.create_prediction_pipeline()
        
        # Generate final report
        report = analyzer.generate_report()
        
        print("\n🎉 Sentiment Analysis Pipeline Completed Successfully!")
        print("=" * 60)
        print("📂 Generated files:")
        print("  - models/ (trained models)")
        print("  - visualizations/ (charts and graphs)")
        print("  - sentiment_analysis_report.md (comprehensive report)")
        
        # Interactive testing
        print("\n🧪 Interactive Testing:")
        while True:
            user_input = input("\nEnter a review to analyze (or 'quit' to exit): ").strip()
            if user_input.lower() == 'quit':
                break
            
            if user_input:
                result = analyzer.predict_sentiment(user_input)
                if result:
                    print(f"Sentiment: {result['sentiment']}")
                    print(f"Confidence: {result['confidence']:.2f}")
                    print("Probabilities:")
                    for sentiment, prob in result['probabilities'].items():
                        print(f"  {sentiment}: {prob:.3f}")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
