"""Tensorflow streamlit page for streamlit app"""

import streamlit as st

st.header("Tensorflow")

st.divider()

st.subheader("Model Loss")
st.write(
    "The goal in training our models for Neural Networks is to \
        decrease the loss as much as possible with every iteration \
            (refered to as an epoch). At the same time we want to \
                increase our accuracy against the test data."
)
st.image("Pictures/tensorflow_Loss.png")
st.caption(
    "Here, we see the loss decrease with every epoch. This shows that \
        the layers we have included in our Neural Network were a good \
            choice and we chose an ideal optimizer function as well \
                as a loss function (probably most importnatly here)."
)

st.divider()

st.subheader("Model Accuracy")
st.write(
    "Another hugely important goal is to increase our accuracy \
        with each Epoch. This shows us the model is learning."
)
st.image("Pictures/tensorflow_Acc.png")
st.caption("Here, we see the accuracy increase steadily with every epoch. ")

st.divider()

st.subheader("About My Model")
st.write("My model Accuracy: 98.03%")

st.write(
    """
The code defines a simple neural network model to classify \
    handwritten digits. The model consists of the following layers:

Flatten: This layer takes the 28x28 pixel input image and transforms \
    it into a one-dimensional array of 784 elements.
Dense (128 units): This is a fully connected layer with 128 neurons. \
    It applies a weighted sum to the input and passes it through the \
        ReLU activation function, introducing non-linearity.
Dropout: This layer randomly deactivates 20% of the neurons during \
    training to prevent overfitting.
Dense (10 units): The final fully connected layer with 10 neurons, \
    one for each digit class. It applies a softmax activation function, \
        producing a probability distribution over the possible output classes.
In essence, the model takes an image, flattens it, processes it \
    through two layers of neurons, and then outputs a probability \
        distribution representing its prediction for which digit \
            the image contains.

The model is compiled with the Adam optimizer, which is known for \
    its efficient training, and the sparse categorical crossentropy \
        loss function, suitable for multi-class classification \
            problems with integer labels. The model's accuracy \
                is tracked during training.

The final test accuracy of **97.91%** indicates that the model \
    performs well on unseen data, successfully classifying \
        handwritten digits with high accuracy.
"""
)
