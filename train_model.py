
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ----------------------------
# Create folders
# ----------------------------

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------

df = pd.read_csv("data/processed_spam.csv")

print(df.head())

# ----------------------------
# Features & Target
# Remove rows with missing values
df = df.dropna(subset=["clean_text"])

# Convert to string (extra safety)
df["clean_text"] = df["clean_text"].astype(str)

X = df["clean_text"]
y = df["label"]

# ----------------------------
# TF-IDF Vectorizer
# ----------------------------

vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(X)

joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

# ----------------------------
# Train-Test Split
# ----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ----------------------------
# Models
# ----------------------------

models = {

    "Naive Bayes":
        MultinomialNB(),

    "SVM":
        LinearSVC(),

    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

}

results = []

best_model = None

best_accuracy = 0

best_name = ""

# ----------------------------
# Training Loop
# ----------------------------

for name, model in models.items():

    print(f"\nTraining {name}")

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    results.append([
        name,
        accuracy,
        precision,
        recall,
        f1
    ])

    joblib.dump(
        model,
        f"models/{name.replace(' ','_')}.pkl"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy

        best_model = model

        best_name = name

# ----------------------------
# Save Best Model
# ----------------------------

joblib.dump(
    best_model,
    "models/best_model.pkl"
)

print("\nBest Model :", best_name)

# ----------------------------
# Save Results
# ----------------------------

result_df = pd.DataFrame(

    results,

    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]

)

print(result_df)

result_df.to_csv(
    "models/model_results.csv",
    index=False
)

# ----------------------------
# Accuracy Comparison
# ----------------------------

plt.figure(figsize=(8, 5))

plt.bar(
    result_df["Model"],
    result_df["Accuracy"]
)

plt.title("Model Accuracy Comparison")

plt.ylabel("Accuracy")

plt.tight_layout()

plt.savefig(
    "images/model_accuracy.png"
)

plt.show()

# ----------------------------
# Confusion Matrix
# ----------------------------

prediction = best_model.predict(X_test)

cm = confusion_matrix(
    y_test,
    prediction
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Best Model Confusion Matrix")

plt.savefig(
    "images/confusion_matrix.png"
)

plt.show()

print("\nTraining Completed Successfully!")
