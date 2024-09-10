"""The details on the imdb model for the streamlit app"""

import streamlit as st

st.header("The IMDB Model for Sentiment Analysis")

st.divider()

st.subheader("About the model:")

st.write(
    """
We used TFIDF (Term Frequency - Inverse Document Frecuency) \
    to take the preprocssed text and convert it into numbers \
        so the machine learning model could read it.

We then used a logistic regression model to train on the \
    new numerical representations of this preprocessed data. 
         
Doing this resulted in an accuracy in predictions for \
    sentiment of **89.32%**.
"""
)

st.image("Pictures/Confusion_Matrix.png")
st.caption(
    "This confusion matrix shows us that the model \
        identified reviews correctly \
            (with it's predictions) 89.32% of the time."
)

st.divider()
