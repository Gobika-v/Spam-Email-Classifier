
import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px

# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(
    page_title="Spam Email Classifier",
    page_icon="📧",
    layout="wide"
)

st.title("📧 Spam Email Classifier")
st.write("Detect whether an email is Spam or Ham using Machine Learning.")

# ----------------------------
# Sidebar
# ----------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dataset",
        "Prediction",
        "Model Performance"
    ]
)

# ----------------------------
# Dataset Page
# ----------------------------

if page == "Dataset":

    st.header("Dataset Preview")

    try:

        df = pd.read_csv("data/processed_spam.csv")

        st.dataframe(df.head(20))

        st.write("Rows :", df.shape[0])
        st.write("Columns :", df.shape[1])

        fig = px.histogram(
            df,
            x="label",
            title="Spam vs Ham Distribution"
        )

        st.plotly_chart(fig, use_container_width=True)

    except:

        st.error("Processed dataset not found.")

# ----------------------------
# Prediction Page
# ----------------------------

elif page == "Prediction":

    st.header("Spam Prediction")

    model_name = st.selectbox(
        "Choose Model",
        [
            "Naive_Bayes",
            "SVM",
            "Logistic_Regression",
            "Random_Forest",
            "best_model"
        ]
    )

    model_path = f"models/{model_name}.pkl"

    if os.path.exists(model_path):

        model = joblib.load(model_path)

        vectorizer = joblib.load(
            "models/tfidf_vectorizer.pkl"
        )

        email = st.text_area(
            "Enter Email",
            height=200
        )

        if st.button("Predict"):

            if email.strip() == "":

                st.warning("Please enter an email.")

            else:

                email_vector = vectorizer.transform(
                    [email]
                )

                prediction = model.predict(
                    email_vector
                )

                if prediction[0] == 1:

                    st.error("🚨 SPAM EMAIL")

                else:

                    st.success("✅ HAM (NOT SPAM)")

                if hasattr(model, "predict_proba"):

                    probability = model.predict_proba(
                        email_vector
                    )

                    st.write(
                        "Spam Probability :",
                        round(probability[0][1] * 100, 2),
                        "%"
                    )

    else:

        st.error(
            "Run train_model.py first."
        )

# ----------------------------
# Model Performance
# ----------------------------

elif page == "Model Performance":

    st.header("Model Comparison")

    try:

        results = pd.read_csv(
            "models/model_results.csv"
        )

        st.dataframe(results)

        fig = px.bar(
            results,
            x="Model",
            y="Accuracy",
            color="Model",
            title="Accuracy Comparison"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        fig2 = px.bar(
            results,
            x="Model",
            y="F1 Score",
            color="Model",
            title="F1 Score Comparison"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    except:

        st.warning(
            "Run train_model.py first."
        )
