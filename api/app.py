#!/usr/bin/env python3
"""
Flask API for Cell Phone Reviews Sentiment Analysis
Provides REST endpoints for sentiment prediction
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
import joblib
import json
import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import logging
from pathlib import Path

# Initialize Flask app
app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentPredictor:
    def __init__(self, model_path="../models"):
        """Initialize the sentiment predictor with trained models"""
        # Handle both relative and absolute paths
        if not Path(model_path).is_absolute():
            # Get the directory of this script and go up one level to find models
            script_dir = Path(__file__).parent
            self.model_path = script_dir / model_path
        else:
            self.model_path = Path(model_path)
        
        # Also try current working directory if relative path doesn't work
        if not self.model_path.exists():
            self.model_path = Path("models")
        self.pipeline = None
        self.label_encoder = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        self.load_models()
    
    def load_models(self):
        """Load trained models and preprocessing components"""
        try:
            # Load prediction pipeline
            pipeline_path = self.model_path / "prediction_pipeline.pkl"
            if pipeline_path.exists():
                self.pipeline = joblib.load(pipeline_path)
                logger.info(" Prediction pipeline loaded successfully")
            else:
                logger.error("Prediction pipeline not found")
                return False
            
            # Load label encoder
            encoder_path = self.model_path / "label_encoder.pkl"
            if encoder_path.exists():
                self.label_encoder = joblib.load(encoder_path)
                logger.info(" Label encoder loaded successfully")
            else:
                logger.error("Label encoder not found")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
            return False
    
    def clean_text(self, text):
        """Clean and preprocess text for prediction"""
        if not text or text.strip() == "":
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
    
    def predict(self, text):
        """Predict sentiment for given text"""
        try:
            if not self.pipeline or not self.label_encoder:
                return {"error": "Models not loaded properly"}
            
            # Clean text
            clean_text = self.clean_text(text)
            
            if not clean_text:
                return {"error": "Empty text after cleaning"}
            
            # Make prediction
            prediction_encoded = self.pipeline.predict([clean_text])[0]
            prediction_proba = self.pipeline.predict_proba([clean_text])[0]
            
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
                'probabilities': confidence_scores,
                'cleaned_text': clean_text
            }
            
        except Exception as e:
            logger.error(f" Prediction error: {str(e)}")
            return {"error": f"Prediction failed: {str(e)}"}

# Initialize predictor
predictor = SentimentPredictor()

@app.route('/')
def home():
    """Home page with API documentation"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models_loaded': predictor.pipeline is not None and predictor.label_encoder is not None
    })

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    Predict sentiment for given text
    
    Request body:
    {
        "text": "This phone case is amazing! Great quality."
    }
    
    Response:
    {
        "sentiment": "positive",
        "confidence": 0.89,
        "probabilities": {
            "positive": 0.89,
            "negative": 0.08,
            "neutral": 0.03
        }
    }
    """
    try:
        # Get request data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form.to_dict()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'Missing required field: text',
                'example': {'text': 'Your review text here'}
            }), 400
        
        text = data['text']
        
        if not text or text.strip() == "":
            return jsonify({
                'error': 'Empty text provided'
            }), 400
        
        # Make prediction
        result = predictor.predict(text)
        
        if 'error' in result:
            return jsonify(result), 500
        
        # Add metadata
        result['timestamp'] = datetime.now().isoformat()
        result['original_text'] = text
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API Error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict sentiment for multiple texts
    
    Request body:
    {
        "texts": [
            "This phone case is amazing!",
            "Terrible quality, broke after one day",
            "It's okay, nothing special"
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'texts' not in data:
            return jsonify({
                'error': 'Missing required field: texts',
                'example': {'texts': ['Review 1', 'Review 2']}
            }), 400
        
        texts = data['texts']
        
        if not isinstance(texts, list):
            return jsonify({
                'error': 'texts must be a list'
            }), 400
        
        if len(texts) > 100:  # Limit batch size
            return jsonify({
                'error': 'Maximum 100 texts per batch'
            }), 400
        
        # Make predictions
        results = []
        for i, text in enumerate(texts):
            if text and text.strip():
                result = predictor.predict(text)
                result['index'] = str(i)
                result['original_text'] = text
                results.append(result)
            else:
                results.append({
                    'index': i,
                    'error': 'Empty text',
                    'original_text': text
                })
        
        return jsonify({
            'predictions': results,
            'count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Batch API Error: {str(e)}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/stats')
def get_stats():
    """Get model and API statistics"""
    return jsonify({
        'model_info': {
            'pipeline_loaded': predictor.pipeline is not None,
            'label_encoder_loaded': predictor.label_encoder is not None,
            'classes': list(predictor.label_encoder.classes_) if predictor.label_encoder else None
        },
        'api_info': {
            'version': '1.0.0',
            'endpoints': ['/predict', '/predict_batch', '/health', '/stats'],
            'max_batch_size': 100
        },
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'available_endpoints': ['/predict', '/predict_batch', '/health', '/stats']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'message': 'Something went wrong on our end'
    }), 500

if __name__ == '__main__':
    # Download required NLTK data
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
    
    # Start the Flask application
    print(" Starting Cell Phone Reviews Sentiment Analysis API")
    print("=" * 60)
    print(" API Endpoints:")
    print("  • POST /predict - Single text prediction")
    print("  • POST /predict_batch - Batch text prediction")  
    print("  • GET /health - Health check")
    print("  • GET /stats - API statistics")
    print("  • GET / - Web interface")
    print("=" * 60)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )
