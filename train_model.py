import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("dataset.csv")

print("Dataset Loaded:")
print(df.head())

# Text column
X = df["text"]

# Label column
y = df["label"]

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TFIDF
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)

# Train model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Save files
joblib.dump(model, "model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
joblib.dump(le, "label_encoder.pkl")

print("Model training completed!") 