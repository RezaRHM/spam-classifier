# Spam Detection using Machine Learning

This project implements a spam detection system using classical Machine Learning techniques.  
The goal is to classify SMS messages as **spam** or **ham (not spam)**.

The project compares two baseline models:

- Multinomial Naive Bayes
- Logistic Regression

---

## 📌 Dataset

The dataset used in this project is the **SMS Spam Collection Dataset** from UCI:

https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection

- ~5,500 labeled SMS messages
- Labels: `spam` and `ham`
- Data is automatically downloaded from the source URL during training

---

## 🧠 Methodology

1. Load dataset from URL
2. Basic data inspection
3. Train/Test split with stratification
4. Feature extraction using **TF-IDF**
5. Model training:
   - Multinomial Naive Bayes
   - Logistic Regression
6. Model evaluation:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion Matrix
7. Threshold tuning for spam probability

---

## 📊 Results (Example)

| Model                  | Accuracy | Spam Recall |
|------------------------|----------|-------------|
| Naive Bayes            | ~0.98    | ~0.85       |
| Logistic Regression    | ~0.98    | Higher / Tunable |

Logistic Regression provides better flexibility through threshold adjustment, allowing control over precision vs recall trade-off.

---

## 📁 Project Structure:


spam-classifier-ml/
├── notebooks/
│   └── spam_classifier.ipynb
├── src/
│   ├── train.py
│   ├── evaluate.py
│   └── utils.py
├── models/
├── requirements.txt
├── README.md
└── .gitignore