# src/utils.py
import io
import zipfile
import requests
import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"

def load_sms_spam_from_url(url: str = DATA_URL) -> pd.DataFrame:
    r = requests.get(url, timeout=30)
    r.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        with z.open("SMSSpamCollection") as f:
            df = pd.read_csv(f, sep="\t", header=None, names=["label", "message"])

    
    df["label"] = df["label"].astype(str).str.strip()
    df["message"] = df["message"].astype(str)

    return df