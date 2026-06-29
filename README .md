📧 Spam Email Classifier

A Machine Learning project that classifies email messages as Spam or Ham (Not Spam) using Natural Language Processing (NLP) and multiple classification algorithms.

🚀 Features
Email text preprocessing
Stopword removal
Stemming
TF-IDF feature extraction
Multiple ML models
Automatic best model selection
Interactive Streamlit web application
Command Line Interface (CLI)
Model comparison
Download prediction results
🛠 Technologies Used
Python
Pandas
NumPy
Scikit-learn
NLTK
Streamlit
Plotly
Matplotlib
Joblib
📂 Project Structure
Spam-Email-Classifier/
│
├── data/
│   ├── spam.csv
│   └── processed_spam.csv
│
├── models/
│   ├── Naive_Bayes.pkl
│   ├── SVM.pkl
│   ├── Logistic_Regression.pkl
│   ├── Random_Forest.pkl
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_results.csv
│
├── images/
│   ├── model_accuracy.png
│   └── confusion_matrix.png
│
├── data_preprocessing.py
├── train_model.py
├── spam_predictor.py
├── app.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
📊 Dataset

The project uses the SMS Spam Collection Dataset.

Columns:

label
text

Target:

ham → 0
spam → 1
🧹 Data Preprocessing

The following preprocessing steps are performed:

Convert text to lowercase
Remove punctuation
Remove numbers
Remove special characters
Tokenization
Remove stopwords
Apply Porter Stemming
🔤 Feature Extraction

Text is converted into numerical features using:

TF-IDF Vectorizer
🤖 Machine Learning Models

The following models are trained and compared:

Multinomial Naive Bayes
Support Vector Machine (LinearSVC)
Logistic Regression
Random Forest Classifier

The model with the highest accuracy is saved automatically as:

models/best_model.pkl
📈 Evaluation Metrics

The models are evaluated using:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
▶ Installation

Clone the repository:

git clone https://github.com/yourusername/Spam-Email-Classifier.git

Move into the project:

cd Spam-Email-Classifier

Install dependencies:

pip install -r requirements.txt
▶ Run the Project
Step 1

Preprocess the dataset

python data_preprocessing.py
Step 2

Train models

python train_model.py
Step 3

Run CLI application

python spam_predictor.py
Step 4

Launch Streamlit

streamlit run app.py