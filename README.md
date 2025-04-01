# 🔗 Phishing URL Detection using RandomForest

## 📌 Project Description
This project focuses on detecting phishing URLs using a **RandomForest** classification model. The model is trained on a dataset containing legitimate and phishing URLs, extracting various features to distinguish between safe and malicious links.

## 🚀 Features
- 🛡 **RandomForest Classifier**: Robust ensemble learning model for classification.
- 🌐 **URL Feature Extraction**: Analyzes lexical, domain, and network-based features.
- 📊 **Performance Evaluation**: Accuracy, precision, recall, and F1-score.
- ⚡ **Fast Predictions**: Classifies URLs in real-time.
- 🔍 **Scalable & Customizable**: Can be extended with more advanced features.

## 🛠 Installation
To set up the project, install the required dependencies:

```bash
pip install pandas scikit-learn numpy requests tldextract
```

## 📂 Dataset
The dataset consists of phishing and legitimate URLs with extracted features. To preprocess and load the dataset:

```python
from data_preprocessing import load_dataset
df = load_dataset("dataset.csv")
print(df.head())
```

## 🎯 Model Training
Train the RandomForest model with:

```bash
python train.py --dataset ./dataset.csv --estimators 100
```

## 🔍 Usage
To predict whether a URL is phishing or legitimate:

```python
from detector import predict_url
url = "http://example.com/login"
result = predict_url(url)
print("Prediction:", result)
```

## 📈 Evaluation
The model's performance is evaluated using:
- ✅ **Accuracy**
- 🔢 **Precision, Recall, and F1-score**

## 🙌 Acknowledgments
- 🔗 **Open-source Phishing Datasets** for training data
- 📊 **Scikit-learn** for machine learning implementation
- 🌍 **Cybersecurity Research Community** for insights on phishing detection
