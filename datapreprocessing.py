import pandas as pd
import string
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK resources (first time only)
nltk.download("stopwords")

# Load dataset
df = pd.read_csv("data/spam.csv", encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]

df.columns = ["label", "text"]

print(df.head())

# Initialize stemmer
stemmer = PorterStemmer()

# Stopwords
stop_words = set(stopwords.words("english"))


def clean_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = text.strip()

    # Tokenization
    words = text.split()

    # Remove stopwords and apply stemming
    cleaned = []

    for word in words:

        if word not in stop_words:

            cleaned.append(stemmer.stem(word))

    return " ".join(cleaned)


df["clean_text"] = df["text"].apply(clean_text)

# Remove empty cleaned emails
df["clean_text"] = df["clean_text"].replace("", pd.NA)

df.dropna(subset=["clean_text"], inplace=True)

# Convert labels to numeric
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# Save processed dataset
df.to_csv("data/processed_spam.csv", index=False)

print("\nDataset Processed Successfully!")

print(df.head())
