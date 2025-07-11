#!/usr/bin/env python3
"""
Project Runner Script
Provides easy commands to run different parts of the sentiment analysis project.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🚀 {description}")
    print(f"Running: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running {description}: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️ {description} interrupted by user")
        return False

def check_requirements():
    """Check if required packages are installed"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        'flask', 'pandas', 'numpy', 'scikit-learn', 
        'nltk', 'matplotlib', 'seaborn', 'wordcloud'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        install_cmd = f"pip install {' '.join(missing_packages)}"
        run_command(install_cmd, "Installing missing packages")
    else:
        print("✅ All required packages are installed!")

def run_api():
    """Run the Flask API server"""
    os.chdir(Path(__file__).parent)
    return run_command("python api/app.py", "Starting Flask API Server")

def run_demo():
    """Run the demo script"""
    os.chdir(Path(__file__).parent)
    return run_command("python demo.py", "Running Demo Script")

def run_jupyter():
    """Open Jupyter notebook"""
    os.chdir(Path(__file__).parent)
    return run_command("jupyter notebook data_exploration.ipynb", "Opening Jupyter Notebook")

def run_tests():
    """Run system tests"""
    os.chdir(Path(__file__).parent)
    return run_command("python test_system.py", "Running System Tests")

def install_requirements():
    """Install project requirements"""
    os.chdir(Path(__file__).parent)
    return run_command("pip install -r requirements.txt", "Installing Requirements")

def download_nltk_data():
    """Download required NLTK data"""
    print("📥 Downloading NLTK data...")
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
        nltk.data.find('taggers/averaged_perceptron_tagger')
        nltk.data.find('corpora/wordnet')
        print("✅ NLTK data already downloaded!")
    except LookupError:
        print("Downloading NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('wordnet')
        print("✅ NLTK data downloaded!")

def main():
    parser = argparse.ArgumentParser(description='Run sentiment analysis project components')
    parser.add_argument('command', choices=[
        'api', 'demo', 'notebook', 'jupyter', 'test', 'install', 'setup', 'all'
    ], help='Command to run')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🎯 SENTIMENT ANALYSIS PROJECT RUNNER")
    print("=" * 60)
    
    if args.command == 'setup':
        print("🔧 Setting up project...")
        install_requirements()
        download_nltk_data()
        print("✅ Project setup complete!")
        
    elif args.command == 'install':
        install_requirements()
        
    elif args.command == 'api':
        check_requirements()
        download_nltk_data()
        print("\n🌐 Starting API server...")
        print("API will be available at: http://127.0.0.1:5000")
        print("Web interface at: http://127.0.0.1:5000")
        print("Press Ctrl+C to stop the server")
        run_api()
        
    elif args.command == 'demo':
        check_requirements()
        download_nltk_data()
        run_demo()
        
    elif args.command in ['notebook', 'jupyter']:
        check_requirements()
        print("\n📊 Opening Jupyter notebook...")
        print("The notebook will open in your default browser")
        run_jupyter()
        
    elif args.command == 'test':
        check_requirements()
        download_nltk_data()
        run_tests()
        
    elif args.command == 'all':
        print("🚀 Running complete project demonstration...")
        install_requirements()
        download_nltk_data()
        run_tests()
        print("\n🎉 All components tested successfully!")
        print("\nNext steps:")
        print("1. Run 'python run_project.py api' to start the web server")
        print("2. Run 'python run_project.py notebook' to explore data")
        print("3. Run 'python run_project.py demo' for quick demo")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("🎯 SENTIMENT ANALYSIS PROJECT RUNNER")
        print("=" * 50)
        print("Available commands:")
        print("  setup    - Install requirements and download NLTK data")
        print("  api      - Start Flask API server with web interface")
        print("  demo     - Run quick demo script")
        print("  notebook - Open Jupyter notebook for data exploration")
        print("  test     - Run system tests")
        print("  install  - Install requirements only")
        print("  all      - Run complete setup and tests")
        print("\nUsage: python run_project.py <command>")
        print("Example: python run_project.py api")
    else:
        main()
