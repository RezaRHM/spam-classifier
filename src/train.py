# src/train.py
import argparse
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from utils import load_sms_spam_from_url, DATA_URL


def main(models_dir: str, test_size: float, random_state: int):
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    df = load_sms_spam_from_url(DATA_URL)
    X_text = df["message"]
    y = df["label"]

    
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X_text, y, test_size=test_size, random_state=random_state, stratify=y
    )

    
    vectorizer = TfidfVectorizer(stop_words="english")
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    
    nb = MultinomialNB()
    nb.fit(X_train, y_train)

    
    lr = LogisticRegression(max_iter=2000)
    lr.fit(X_train, y_train)

    
    joblib.dump(vectorizer, f"{models_dir}/vectorizer.joblib")
    joblib.dump(nb, f"{models_dir}/nb_model.joblib")
    joblib.dump(lr, f"{models_dir}/lr_model.joblib")

    
    joblib.dump((X_test_text, y_test), f"{models_dir}/test_set.joblib")

    print("Saved:")
    print(f"- {models_dir}/vectorizer.joblib")
    print(f"- {models_dir}/nb_model.joblib")
    print(f"- {models_dir}/lr_model.joblib")
    print(f"- {models_dir}/test_set.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    main(args.models_dir, args.test_size, args.random_state)