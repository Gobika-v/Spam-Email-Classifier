import joblib

# ----------------------------
# Load Model & Vectorizer
# ----------------------------

try:
    model = joblib.load("models/best_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

except FileNotFoundError:

    print("Error!")
    print("Run train_model.py first.")
    exit()

# ----------------------------
# Title
# ----------------------------

print("=" * 60)
print("         SPAM EMAIL CLASSIFIER")
print("=" * 60)

# ----------------------------
# User Input
# ----------------------------

email = input("\nEnter Email Message:\n\n")

# ----------------------------
# Transform Input
# ----------------------------

email_vector = vectorizer.transform([email])

# ----------------------------
# Prediction
# ----------------------------

prediction = model.predict(email_vector)

# ----------------------------
# Probability (if supported)
# ----------------------------

if hasattr(model, "predict_proba"):

    probability = model.predict_proba(email_vector)

    spam_probability = probability[0][1]

else:

    spam_probability = None

# ----------------------------
# Output
# ----------------------------

print("\n" + "=" * 60)

if prediction[0] == 1:

    print("Prediction : SPAM")

else:

    print("Prediction : NOT SPAM (HAM)")

if spam_probability is not None:

    print(f"Spam Probability : {spam_probability:.2%}")

print("=" * 60)
