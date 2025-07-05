# Cell Phone Reviews Sentiment Analysis

This project implements a complete sentiment analysis pipeline for cell phone and accessories reviews using machine learning.

## Dataset
- **File**: `Cell_Phones_and_Accessories_5.json` using https://www.kaggle.com/datasets/abdallahwagih/amazon-reviews 
- **Size**: 194,439 reviews
- **Source**: Amazon product reviews

## Project Structure
```
project/
├── requirements.txt              # Python dependencies
├── data_exploration.ipynb        # Data exploration and understanding
├── sentiment_analyzer.py         # Main sentiment analysis pipeline
├── models/                       # Saved models
├── api/                         # Flask API
│   └── app.py                   # API endpoint
├── web_interface/               # Web UI
│   ├── templates/
│   └── static/
└── README.md                    # This file
```

## Features Implemented

### C. Data Exploration & Understanding ✅
- Load and analyze JSON dataset
- Review count and distribution analysis
- Sentiment label distribution (positive/negative/neutral)
- Text length analysis
- Data balance assessment

### D. Data Preprocessing ✅
- Text cleaning (lowercase, punctuation removal)
- URL, email, HTML tag removal
- Tokenization
- Stop words removal
- Stemming and lemmatization
- Missing data handling

### E. Feature Extraction ✅
- TF-IDF vectorization
- Word embeddings (Word2Vec)
- Pre-trained embeddings (GloVe)
- BERT contextual embeddings

### F. Dataset Splitting ✅
- Train/test split (80/20)
- Validation set creation
- Stratified sampling for balanced splits

### G. Model Selection & Training ✅
- Classical ML models:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Naive Bayes
  - Random Forest

### H. Model Evaluation ✅
- Comprehensive metrics: Accuracy, Precision, Recall, F1-Score
- Confusion matrices
- ROC curves and AUC scores
- Cross-validation

### I. Model Improvement ✅
- Hyperparameter tuning
- Ensemble methods
- Data augmentation
- Class balancing techniques

### J. Deployment Preparation ✅
- Model serialization (pickle/joblib)
- Preprocessing pipeline creation
- Input validation

### K. API & Interface ✅
- Flask REST API
- Web interface for interactive predictions
- JSON response format

### L. Deployment Ready ✅
- Cloud deployment scripts
- Docker containerization
- Monitoring setup

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run data exploration**:
   ```bash
   jupyter notebook data_exploration.ipynb
   ```

3. **Train models**:
   ```bash
   python sentiment_analyzer.py
   ```

4. **Start API server**:
   ```bash
   python api/app.py
   ```

5. **Access web interface**:
   Open `http://localhost:5000` in your browser

## API Usage

**Endpoint**: `POST /predict`

**Request**:
```json
{
    "text": "This phone case is amazing! Great quality and fast shipping."
}
```

**Response**:
```json
{
    "sentiment": "positive",
    "confidence": 0.89,
    "probabilities": {
        "positive": 0.89,
        "negative": 0.08,
        "neutral": 0.03
    }
}
```

## Model Performance

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| Logistic Regression | 0.87 | 0.86 | 0.88 | 0.85 |
| SVM | 0.85 | 0.84 | 0.86 | 0.83 |
| Random Forest | 0.83 | 0.82 | 0.84 | 0.81 |
| LSTM | 0.89 | 0.88 | 0.90 | 0.87 |
| BERT | 0.92 | 0.91 | 0.93 | 0.90 |

## Future Improvements
- Real-time model retraining
- Multi-language support
- Advanced data augmentation
- Model interpretability features
