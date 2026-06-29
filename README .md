# 📧 Spam Email Classifier

A Machine Learning project that classifies email messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and multiple Machine Learning algorithms.

---

## 🚀 Features

- 📧 Email text preprocessing
- 🔤 Tokenization
- 🚫 Stopword removal
- ✂️ Stemming using Porter Stemmer
- 📊 TF-IDF feature extraction
- 🤖 Multiple Machine Learning models
- 🏆 Automatic best model selection
- 💻 Command Line Interface (CLI)
- 🌐 Interactive Streamlit Web Application
- 📈 Model comparison
- 💾 Save and load trained models using Joblib

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Streamlit
- Matplotlib
- Plotly
- Joblib

---

## 📂 Project Structure

```text
Spam-Email-Classifier/
│
├── data/
│   ├── spam.csv
│   └── processed_spam.csv
│
├── images/
│
├── models/
│
├── app.py
├── data_preprocessing.py
├── train_model.py
├── spam_predictor.py
├── utils.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

This project uses the **SMS Spam Collection Dataset**.

**Columns**

- label
- text

**Target**

| Label | Value |
|-------|-------|
| Ham | 0 |
| Spam | 1 |

---

## 🧹 Data Preprocessing

The following preprocessing steps are performed:

- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Remove special characters
- Tokenization
- Stopword removal
- Porter Stemming

---

## 🔤 Feature Extraction

The cleaned email text is converted into numerical features using:

- TF-IDF Vectorizer

---

## 🤖 Machine Learning Models

The following models are trained and compared:

- Multinomial Naive Bayes
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest Classifier

The best-performing model is automatically saved.

---

## 📈 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

---



## ▶️ How to Run

### 1️⃣ Preprocess the Dataset

```bash
python data_preprocessing.py
```

### 2️⃣ Train the Models

```bash
python train_model.py
```

### 3️⃣ Run the CLI Application

```bash
python spam_predictor.py
```

### 4️⃣ Launch the Streamlit Web App

```bash
streamlit run app.py
```

## 📌 Future Improvements

- Use a larger email dataset
- Deploy the application online
- Add BERT-based text classification
- Integrate Gmail API
- Detect phishing emails
- Deep Learning using LSTM

---

## 🎯 Skills Demonstrated

- Natural Language Processing (NLP)
- Text Preprocessing
- TF-IDF Vectorization
- Binary Classification
- Machine Learning Model Comparison
- Streamlit Deployment
- Model Serialization using Joblib

---
