"""
Preprocessing Utilities — Text cleaning, TF-IDF, Scaling, Feature Fusion
"""
import re
import string
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def build_features(text, numeric_data):
    cleaned = clean_text(text)
    return cleaned, numeric_data


PHYSIO_FEATURES = [
    "heart_rate", "sleep_hours", "study_hours_per_day",
    "days_since_last_break", "social_interactions_per_week",
    "water_intake_liters", "exercise_minutes_per_day",
]

def clean_text(text: str) -> str:
    """Lowercase → remove punctuation → remove stopwords → stem."""
    text = text.lower()
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return " ".join(tokens)

def build_features(df, tfidf=None, scaler=None, fit=False):
    """
    Returns combined sparse feature matrix and fitted transformers.
    If fit=True: fits tfidf & scaler on df (training mode)
    If fit=False: transforms using pre-fitted tfidf & scaler (inference mode)
    """
    # ── Text features ──────────────────────────────────────────────────────
    cleaned = df["text"].apply(clean_text)
    if fit:
        tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        text_features = tfidf.fit_transform(cleaned)
    else:
        text_features = tfidf.transform(cleaned)

    # ── Numerical features ─────────────────────────────────────────────────
    num_data = df[PHYSIO_FEATURES].fillna(df[PHYSIO_FEATURES].median())
    if fit:
        scaler = StandardScaler()
        num_features = scaler.fit_transform(num_data)
    else:
        num_features = scaler.transform(num_data)

    # ── Combine sparse + dense ─────────────────────────────────────────────
    num_sparse = sp.csr_matrix(num_features)
    combined = sp.hstack([text_features, num_sparse])

    return combined, tfidf, scaler