import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


# -----------------------------------
# LOAD DATASET
# -----------------------------------

data = pd.read_csv("dataset.csv")

print("Dataset loaded successfully!")
print("Total reviews:", len(data))


# -----------------------------------
# INPUT AND OUTPUT
# -----------------------------------

X = data["review"]
y = data["sentiment"]


# -----------------------------------
# SPLIT DATA
# -----------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# -----------------------------------
# CREATE MODEL
# -----------------------------------

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression()
    )
])


# -----------------------------------
# TRAIN MODEL
# -----------------------------------

print("\nTraining model...")

model.fit(X_train, y_train)

print("Training completed!")


# -----------------------------------
# TEST MODEL
# -----------------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)


print("\n--------------------------------")
print("MODEL RESULTS")
print("--------------------------------")

print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# -----------------------------------
# SAVE MODEL
# -----------------------------------

joblib.dump(model, "sentiment_model.pkl")

print("\nModel saved successfully!")
print("File: sentiment_model.pkl")