# src/evaluate.py
import argparse
import joblib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay


def eval_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== {name} ===")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred, digits=3))

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.show()


def eval_threshold(name, model, X_test, y_test, thresholds):
    if not hasattr(model, "predict_proba"):
        print(f"{name} does not support predict_proba")
        return

    proba = model.predict_proba(X_test)
    spam_idx = np.where(model.classes_ == "spam")[0][0]
    spam_proba = proba[:, spam_idx]

    print(f"\n--- Threshold tuning: {name} ---")
    for t in thresholds:
        y_pred_t = np.where(spam_proba >= t, "spam", "ham")
        print(f"\nThreshold: {t}")
        print("Accuracy:", accuracy_score(y_test, y_pred_t))
        print(classification_report(y_test, y_pred_t, digits=3))


def main(models_dir: str):
    vectorizer = joblib.load(f"{models_dir}/vectorizer.joblib")
    nb = joblib.load(f"{models_dir}/nb_model.joblib")
    lr = joblib.load(f"{models_dir}/lr_model.joblib")
    X_test_text, y_test = joblib.load(f"{models_dir}/test_set.joblib")

    X_test = vectorizer.transform(X_test_text)

    eval_model("MultinomialNB", nb, X_test, y_test)
    eval_model("LogisticRegression", lr, X_test, y_test)

    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    eval_threshold("MultinomialNB", nb, X_test, y_test, thresholds)
    eval_threshold("LogisticRegression", lr, X_test, y_test, thresholds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str, default="models")
    args = parser.parse_args()
    main(args.models_dir)