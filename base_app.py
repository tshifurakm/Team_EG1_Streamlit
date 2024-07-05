"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: ExploreAI Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""

import os

import joblib

# Data dependencies
import pandas as pd

# Streamlit dependencies
import streamlit as st

# Vectorizer
news_vectorizer = open("Vectorizer/count_vectorizer.pkl", "rb")
test_cv = joblib.load(news_vectorizer)  # loading your vectorizer from the pkl file

# Load your raw data
# raw = pd.read_csv("streamlit/train.csv")

# Define constants
label_to_category = {
    0: "Business",
    1: "Education",
    2: "Entertainment",
    3: "Technology",
    4: "Sports",
}

models = {
    "Logistic Regression": "Models/logreg_model.pkl",
    "Random Forest": "Models/forest_model.pkl",
    "SVC": "Models/svc_model.pkl",
}

# Define css formatting
st.markdown(
    """
    <style>
    .title {
        # text-align: center;
        font-size: 32px;
        font-weight: bold;
        color: #2E86C1;
    }
    .header {
        font-size: 24px;
        margin-top: 20px;
        color: #2E86C1;
    }
    .content {
        font-size: 18px;
        line-height: 1.6;
        
    }
    .subheader {
        font-size: 20px;
        color: #FF5733;
        margin-top: 20px;
    }
    .contact {
        font-size: 18px;
        color: #28B463;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# The main function where we will build the actual app
def main():
    """Frontier Times News Classifier"""

    # Creates a main title and subheader on the page
    st.markdown(
        """
    <h1 style="color:#FF5733;">Frontier Times News Classifier App</h1>
    <p>This application uses Machine Learning models to classify news articles.</p>
    <hr>
    """,
        unsafe_allow_html=True,
    )

    # Creating sidebar with selection box -
    # you can create multiple pages this way
    options = ["Prediction", "Information"]
    selection = st.sidebar.selectbox("Choose Option", options)

    # Building out the "Information" page
    if selection == "Information":
        # Markdown for information page
        st.markdown(
            '<div class="title">About The Project</div>', unsafe_allow_html=True
        )
        st.markdown(
            """We are excited to present NewsIQ Classifier, a state-of-the-art tool developed by DataMinds
			   to enhance the categorization of news articles for Frontier Times. This project employs advanced machine learning
			   techniques to ensure precise and efficient classification of news content, streamlining operations and enhancing the
			   user experience."""
        )
        st.markdown(
            '<div class="header">Project Overview</div>', unsafe_allow_html=True
        )
        st.markdown(
            """<div class="content">Introduction: This project is designed to 
			  classify news articles using machine learning models.<br>Purpose: To provide 
			  an easy-to-use tool for categorizing news articles into predefined categories.<br>Usage: 
			  Useful for journalists, researchers, and anyone interested in news classification.</div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="header">How It Works</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="content">Workflow:<br>1. User inputs text<br>2. Text is processed and 
			  transformed<br>3. Model predicts the category<br>4. Result is displayed to the user<br><br>Models 
			  Used: Logistic Regression, Random Forest, etc.<br>Data: The dataset includes news articles from 
			  various sources.</div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="header">Features</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="content">- Interactive Text Input: Users can input their text for 
			  classification.<br>- Model Selection: Users can choose from multiple models.<br>- 
			  Real-time Predictions: Get instant classification results.<br>- Visualizations: 
			  Includes EDA visualizations to understand the data better.</div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="header">Instructions</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="content">1. Navigate to the Prediction page.<br>2. Enter your news 
			  article text in the provided text area.<br>3. Select the model you want to use for prediction.<br>4. 
			  Click the \'Classify\' button to get the results.</div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="header">About the Team</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="content">- Sifundo : Team Lead<br>- Megan: Project Manager<br>- 
			  Simphiwe: GitHub Manager<br>- Ntembeko: Streamlit Manager<br>- Khuthadzo: GitHub Manager<br>- 
			  Lungile: Design Lead<br>Contact: You can reach us at <a href="info@dataminds.com" 
			  class="contact">email@example.com</a>.</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="header">Technical Details</div>', unsafe_allow_html=True
        )
        st.markdown(
            """<div class="content">Tech Stack:<br>- Python<br>- Streamlit<br>- Scikit-learn<br>
			  Deployment: Hosted on Streamlit Cloud</div>""",
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="header">References and Further Reading</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            """<div class="content">- <a href="https://docs.streamlit.io/" target="_blank">Streamlit
			   Documentation</a><br>- <a href="https://scikit-learn.org/stable/documentation.html" 
			   target="_blank">Scikit-learn Documentation</a></div>""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="header">Future Work</div>', unsafe_allow_html=True)
        st.markdown(
            """<div class="content">Improvements:<br>- Add more models<br>- Improve UI/UX<br>- Further refine training datasets<br>- Enhance prediction accuracy</div>""",
            unsafe_allow_html=True,
        )

    # Building out the prediction page
    if selection == "Prediction":
        st.info("Prediction with ML Models")
        # Creating a text box for user input
        news_text = st.text_area("Enter Text", "Type Here")

        # Choose a model of your choice
        model_choice = st.selectbox("Choose a model to evaluate", list(models.keys()))

        if st.button("Classify"):
            # Transforming user input with vectorizer
            vect_text = test_cv.transform([news_text]).toarray()
            predictor_path = models[model_choice]
            # Load your .pkl file with the model of your choice + make predictions

            predictor = joblib.load(open(predictor_path, "rb"))
            prediction = predictor.predict(vect_text)
            category = label_to_category[prediction[0]]

            # When model has successfully run, will print prediction
            st.success(f"Text Categorized as: **{category}**")


# Required to let Streamlit instantiate our web app.
if __name__ == "__main__":
    main()
