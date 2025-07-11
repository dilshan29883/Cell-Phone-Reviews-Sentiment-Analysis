#!/usr/bin/env python3
"""
Setup and Installation Script for Cell Phone Reviews Sentiment Analysis
Handles environment setup, dependencies, and model training
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"🔧 {description}...")
    try:
        if platform.system() == "Windows":
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=True, capture_output=True, text=True)
        
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error during {description}: {e}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    # Upgrade pip first
    if not run_command(f"{sys.executable} -m pip install --upgrade pip", "Upgrading pip"):
        return False
    
    # Install requirements
    if not run_command(f"{sys.executable} -m pip install -r requirements.txt", "Installing requirements"):
        return False
    
    return True

def download_nltk_data():
    """Download required NLTK datasets"""
    print("📚 Downloading NLTK data...")
    
    import nltk
    
    datasets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']
    
    for dataset in datasets:
        try:
            nltk.data.find(f'tokenizers/{dataset}' if dataset == 'punkt' else f'corpora/{dataset}')
            print(f"✅ {dataset} already downloaded")
        except LookupError:
            print(f"📥 Downloading {dataset}...")
            nltk.download(dataset, quiet=True)
            print(f"✅ {dataset} downloaded successfully")
    
    return True

def create_directories():
    """Create necessary directories"""
    print("📁 Creating project directories...")
    
    directories = ['models', 'visualizations', 'api/static', 'data']
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def check_dataset():
    """Check if dataset file exists"""
    print("📊 Checking dataset availability...")
    
    dataset_file = "Cell_Phones_and_Accessories_5.json"
    
    if Path(dataset_file).exists():
        file_size = Path(dataset_file).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Dataset found: {dataset_file} ({file_size:.2f} MB)")
        return True
    else:
        print(f"❌ Dataset not found: {dataset_file}")
        print("📥 Please download the dataset and place it in the project directory")
        print("💡 You can use the Amazon review datasets from: https://nijianmo.github.io/amazon/index.html")
        return False

def train_models():
    """Train the sentiment analysis models"""
    print("🤖 Training sentiment analysis models...")
    
    if not Path("Cell_Phones_and_Accessories_5.json").exists():
        print("❌ Dataset not found. Skipping model training.")
        return False
    
    # Run the main sentiment analysis script
    if not run_command(f"{sys.executable} sentiment_analyzer.py", "Training models"):
        print("❌ Model training failed")
        return False
    
    # Check if models were created
    model_files = ['models/prediction_pipeline.pkl', 'models/label_encoder.pkl']
    
    for model_file in model_files:
        if Path(model_file).exists():
            print(f"✅ Model created: {model_file}")
        else:
            print(f"❌ Model not found: {model_file}")
            return False
    
    return True

def test_api():
    """Test the API functionality"""
    print("🧪 Testing API functionality...")
    
    # Import API components
    try:
        sys.path.append('api')
        from app import predictor
        
        # Test prediction
        test_text = "This phone case is amazing! Great quality and fast shipping."
        result = predictor.predict(test_text)
        
        if 'error' in result:
            print(f"❌ API test failed: {result['error']}")
            return False
        
        print(f"✅ API test successful!")
        print(f"   Text: '{test_text[:50]}...'")
        print(f"   Predicted sentiment: {result['sentiment']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ API test failed: {e}")
        return False

def print_usage_instructions():
    """Print usage instructions"""
    print("\n🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    print("\n🚀 HOW TO USE:")
    print("-" * 30)
    
    print("\n1. 📊 Explore the data:")
    print("   jupyter notebook data_exploration.ipynb")
    
    print("\n2. 🤖 Train models (if not done automatically):")
    print("   python sentiment_analyzer.py")
    
    print("\n3. 🌐 Start the API server:")
    if platform.system() == "Windows":
        print("   cd api")
        print("   python app.py")
    else:
        print("   cd api && python app.py")
    
    print("\n4. 🔗 Access the web interface:")
    print("   Open http://localhost:5000 in your browser")
    
    print("\n📡 API ENDPOINTS:")
    print("-" * 30)
    print("   • POST /predict - Single text prediction")
    print("   • POST /predict_batch - Batch predictions")
    print("   • GET /health - Health check")
    print("   • GET /stats - API statistics")
    
    print("\n🧪 EXAMPLE API USAGE:")
    print("-" * 30)
    print("   curl -X POST http://localhost:5000/predict \\")
    print("        -H 'Content-Type: application/json' \\")
    print("        -d '{\"text\": \"This phone case is amazing!\"}'")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("-" * 30)
    print("   ├── sentiment_analyzer.py     # Main pipeline")
    print("   ├── data_exploration.ipynb    # Jupyter notebook")
    print("   ├── api/app.py                # Flask API")
    print("   ├── models/                   # Trained models")
    print("   ├── visualizations/           # Generated charts")
    print("   └── requirements.txt          # Dependencies")

def main():
    """Main setup function"""
    print("🚀 CELL PHONE REVIEWS SENTIMENT ANALYSIS SETUP")
    print("=" * 60)
    print("This script will set up the complete sentiment analysis pipeline")
    print()
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Failed to install dependencies")
        sys.exit(1)
    
    # Download NLTK data
    if not download_nltk_data():
        print("❌ Failed to download NLTK data")
        sys.exit(1)
    
    # Check dataset
    dataset_available = check_dataset()
    
    # Train models if dataset is available
    if dataset_available:
        if not train_models():
            print("⚠️  Model training failed, but you can still use the API with pre-trained models")
    else:
        print("⚠️  Skipping model training due to missing dataset")
    
    # Test API (only if models exist)
    if Path("models/prediction_pipeline.pkl").exists():
        if not test_api():
            print("⚠️  API test failed, but setup is otherwise complete")
    
    # Print usage instructions
    print_usage_instructions()

if __name__ == "__main__":
    main()
