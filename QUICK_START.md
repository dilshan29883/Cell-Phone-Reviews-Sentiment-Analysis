# 🚀 Quick Start Guide

## How to Run Your Sentiment Analysis Project

Your sentiment analysis project is now complete! Here are the different ways to run it:

### 🔧 Initial Setup (Run this first)
```bash
# Option 1: Using Python script
python run_project.py setup

# Option 2: Using batch file (Windows)
run_project.bat setup

# Option 3: Manual setup
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger'); nltk.download('wordnet')"
```

### 🌐 Start Web Interface (Recommended)
```bash
# Option 1: Using Python script
python run_project.py api

# Option 2: Using batch file (Windows)
run_project.bat api

# Option 3: Direct command
python api/app.py
```
- **Access at**: http://127.0.0.1:5000
- **Features**: Web form for sentiment analysis, batch processing, API endpoints

### 📊 Explore Data with Jupyter Notebook
```bash
# Option 1: Using runner script
python run_project.py notebook

# Option 2: Direct command
jupyter notebook data_exploration.ipynb
```
- **Features**: Data exploration, model training, visualization

### 🎯 Quick Demo
```bash
# Option 1: Using runner script
python run_project.py demo

# Option 2: Direct command
python demo.py
```
- **Features**: Command-line sentiment analysis examples

### 🧪 Run Tests
```bash
# Option 1: Using runner script
python run_project.py test

# Option 2: Direct command
python test_system.py
```
- **Features**: Complete system validation

## 📋 Available Commands Summary

| Command | Description | Access URL |
|---------|-------------|------------|
| `setup` | Install requirements & NLTK data | - |
| `api` | Start web server | http://127.0.0.1:5000 |
| `demo` | Run CLI demo | Terminal output |
| `notebook` | Open Jupyter notebook | Browser |
| `test` | Run system tests | Terminal output |

## 🔗 API Endpoints

When running the API server, these endpoints are available:

- **GET** `/` - Web interface
- **GET** `/health` - Health check
- **POST** `/predict` - Single text prediction
- **POST** `/predict_batch` - Batch predictions

### Example API Usage:
```bash
# Health check
curl http://127.0.0.1:5000/health

# Single prediction
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This phone is amazing!"}'

# Batch prediction
curl -X POST http://127.0.0.1:5000/predict_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Great product!", "Terrible quality"]}'
```

## 🎉 Project Features

✅ **Complete Pipeline**: Data loading → Preprocessing → Model training → Evaluation
✅ **Web Interface**: User-friendly HTML form for sentiment analysis  
✅ **REST API**: JSON endpoints for integration
✅ **Jupyter Notebook**: Interactive data exploration and model training
✅ **Multiple Models**: Logistic Regression, Random Forest, SVM comparison
✅ **Comprehensive Testing**: Automated system validation
✅ **Docker Support**: Ready for containerized deployment
✅ **Production Ready**: Error handling, logging, documentation

## 🛠️ Troubleshooting

### Common Issues:

1. **Missing packages**: Run `python run_project.py setup`
2. **NLTK data missing**: The setup command downloads required data
3. **Port already in use**: Check if another Flask app is running on port 5000
4. **Permission errors**: Make sure you have write access to the project directory

### Getting Help:

- Check `README.md` for detailed documentation
- Review `PROJECT_COMPLETION_REPORT.md` for project overview
- Run `python run_project.py test` to validate everything is working

---

**Ready to start?** Run `python run_project.py setup` then `python run_project.py api` to get started! 🎯
