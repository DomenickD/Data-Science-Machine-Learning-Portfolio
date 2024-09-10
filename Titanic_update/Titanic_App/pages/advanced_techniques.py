"""the advanced techniques streamlit page"""

import streamlit as st

st.header("Advanced Display")

st.divider()

st.caption(
    "This page will have techniques that were learned but not \
        benefitial to the Titanic Dataset."
)
st.divider()

st.subheader("Ensemble Learning - Boosting")

st.write(
    """

Sequentially trains models, focusing on correcting errors made \
    by previous models to improve overall accuracy. 
         
Using this module with 100 estimators at a learning rate of 0.1, \
    our boosting model obtained an accuracy of *89.00%*. 
"""
)

st.divider()

st.subheader("Ensemble Learning - Bagging")

st.write(
    """

Averages predictions from multiple decision trees trained on random \
    subsets of data to reduce overfitting.

Using this model with a Logistic Regression model pipeline, we had \
    10 estimators. This gave us an accuracy of *93.06%*.
"""
)

st.divider()

st.subheader("Ensemble Learning - Stacking")

st.write(
    """

Combines predictions from different models (e.g., linear regression, \
    decision trees, neural networks) using another model to learn how \
        to best weigh their outputs.
         
To build the stacking model, I used SVC, NB, and a Random Forest \
    Classifier. I then designated the final estimator as Logistic \
        Regression (because this was my highest rated raw model).

The accuracy for the stacking model compliation is: *84.00%*.
"""
)

st.divider()

st.subheader("Ensemble Learning - Voting")

st.write(
    """

Makes predictions based on the most frequent class label predicted \
    by a set of models. (Much like Random Forests.)
         
We used a hard voting method here because it is a classification \
    problem and obtained an accuracy of *85.00%*.
"""
)
