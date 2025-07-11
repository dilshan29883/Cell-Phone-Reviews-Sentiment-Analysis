#!/usr/bin/env python3
"""
Comprehensive System Test for Cell Phone Reviews Sentiment Analysis
Tests all components to ensure everything is working correctly
"""

import os
import sys
import json
import requests
import joblib
import pandas as pd
from pathlib import Path
import time

def test_header():
    """Print test header"""
    print("🧪 COMPREHENSIVE SYSTEM TEST")
    print("=" * 60)
    print("Testing all components of the sentiment analysis system")
    print("=" * 60)

def test_file_structure():
    """Test if all required files exist"""
    print("\n📁 TESTING FILE STRUCTURE")
    print("-" * 40)
    
    required_files = [
        "sentiment_analyzer.py",
        "demo.py", 
        "requirements.txt",
        "README.md",
        "PROJECT_COMPLETION_REPORT.md",
        "data_exploration.ipynb",
        "Dockerfile",
        "docker-compose.yml",
        "Cell_Phones_and_Accessories_5.json",
        "api/app.py",
        "api/templates/index.html",
        "models/prediction_pipeline.pkl",
        "models/label_encoder.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {len(missing_files)}")
        return False
    else:
        print(f"\n✅ All {len(required_files)} required files present")
        return True

def test_dataset():
    """Test dataset accessibility and basic properties"""
    print("\n📊 TESTING DATASET")
    print("-" * 40)
    
    try:
        dataset_path = "Cell_Phones_and_Accessories_5.json"
        
        if not Path(dataset_path).exists():
            print("❌ Dataset file not found")
            return False
        
        # Get file size
        file_size = Path(dataset_path).stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Dataset file found ({file_size:.1f} MB)")
        
        # Test loading a few lines
        line_count = 0
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 10:  # Test first 10 lines
                    break
                try:
                    review = json.loads(line.strip())
                    line_count += 1
                except json.JSONDecodeError:
                    print(f"⚠️  Invalid JSON on line {i+1}")
        
        print(f"✅ Successfully parsed {line_count} sample reviews")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_models():
    """Test if trained models can be loaded and used"""
    print("\n🤖 TESTING TRAINED MODELS")
    print("-" * 40)
    
    try:
        # Load pipeline
        pipeline = joblib.load('models/prediction_pipeline.pkl')
        print("✅ Prediction pipeline loaded")
        
        # Load encoder
        encoder = joblib.load('models/label_encoder.pkl')
        print("✅ Label encoder loaded")
        print(f"✅ Available classes: {list(encoder.classes_)}")
        
        # Test prediction
        test_text = "This phone case is amazing!"
        prediction = pipeline.predict([test_text])[0]
        probabilities = pipeline.predict_proba([test_text])[0]
        sentiment = encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        print(f"✅ Test prediction successful:")
        print(f"   Text: '{test_text}'")
        print(f"   Sentiment: {sentiment}")
        print(f"   Confidence: {confidence:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_api_endpoints():
    """Test API endpoints"""
    print("\n🌐 TESTING API ENDPOINTS")
    print("-" * 40)
    
    base_url = "http://127.0.0.1:5000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health endpoint: {data['status']}")
            print(f"✅ Models loaded: {data['models_loaded']}")
        else:
            print(f"❌ Health endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health endpoint unreachable: {e}")
        return False
    
    # Test stats endpoint
    try:
        response = requests.get(f"{base_url}/stats", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats endpoint working")
            print(f"✅ API version: {data['api_info']['version']}")
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Stats endpoint error: {e}")
    
    # Test prediction endpoint
    try:
        test_payload = {"text": "This phone case is amazing! Great quality."}
        response = requests.post(f"{base_url}/predict", json=test_payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction endpoint working")
            print(f"   Predicted: {data['sentiment']} (confidence: {data['confidence']:.3f})")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction endpoint error: {e}")
        return False
    
    # Test batch prediction endpoint
    try:
        batch_payload = {
            "texts": [
                "Amazing product! Love it!",
                "Terrible quality, don't buy",
                "It's okay, nothing special"
            ]
        }
        response = requests.post(f"{base_url}/predict_batch", json=batch_payload, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction endpoint working")
            print(f"   Processed {data['count']} texts")
            for i, pred in enumerate(data['predictions']):
                print(f"   {i+1}. {pred['sentiment']} (conf: {pred['confidence']:.3f})")
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Batch prediction error: {e}")
        return False
    
    return True

def test_web_interface():
    """Test web interface accessibility"""
    print("\n🌐 TESTING WEB INTERFACE")
    print("-" * 40)
    
    try:
        response = requests.get("http://127.0.0.1:5000", timeout=5)
        if response.status_code == 200:
            content = response.text
            if "Cell Phone Reviews Sentiment Analysis" in content:
                print("✅ Web interface accessible")
                print("✅ Page content loads correctly")
                if "Try the Sentiment Analyzer" in content:
                    print("✅ Interactive demo section present")
                if "API Documentation" in content:
                    print("✅ API documentation section present")
                return True
            else:
                print("❌ Web interface content incorrect")
                return False
        else:
            print(f"❌ Web interface failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Web interface unreachable: {e}")
        return False

def test_performance():
    """Test system performance"""
    print("\n⚡ TESTING PERFORMANCE")
    print("-" * 40)
    
    test_texts = [
        "This phone case is amazing! Great quality and fast shipping.",
        "Terrible product. Broke after one day. Don't waste your money.",
        "It's okay. Nothing special but does the job.",
        "Love the design and functionality. Highly recommend!",
        "Poor quality and expensive. Not worth the money."
    ]
    
    try:
        # Test single predictions
        start_time = time.time()
        for text in test_texts:
            response = requests.post(
                "http://127.0.0.1:5000/predict", 
                json={"text": text}, 
                timeout=10
            )
            if response.status_code != 200:
                print(f"❌ Performance test failed on: '{text[:30]}...'")
                return False
        
        single_time = time.time() - start_time
        avg_single_time = (single_time / len(test_texts)) * 1000  # ms
        
        print(f"✅ Single predictions: {avg_single_time:.1f}ms average")
        
        # Test batch prediction
        start_time = time.time()
        response = requests.post(
            "http://127.0.0.1:5000/predict_batch",
            json={"texts": test_texts},
            timeout=15
        )
        batch_time = (time.time() - start_time) * 1000  # ms
        
        if response.status_code == 200:
            print(f"✅ Batch prediction: {batch_time:.1f}ms for {len(test_texts)} texts")
            print(f"✅ Batch efficiency: {batch_time/len(test_texts):.1f}ms per text")
        else:
            print("❌ Batch performance test failed")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests"""
    test_header()
    
    tests = [
        ("File Structure", test_file_structure),
        ("Dataset", test_dataset),
        ("Models", test_models),
        ("API Endpoints", test_api_endpoints),
        ("Web Interface", test_web_interface),
        ("Performance", test_performance)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n🏆 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<20} {status}")
    
    print("-" * 60)
    print(f"OVERALL: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is fully operational! 🎉")
        print("\n🚀 READY FOR PRODUCTION USE!")
        print("\n💡 Access the system at: http://127.0.0.1:5000")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
