"""Pytorch streamlit page info summary"""

import streamlit as st

st.header("Pytorch")

st.divider()

st.subheader("Loss Accuracy Graphs")
st.image("Pictures/NN-loss-acc.png")
st.caption(
    "On the left, we see the loss per epoch. \n\
        On the right, we see the accuaracy increase with each epoch."
)

st.divider()

st.subheader("About My Model")
st.write("My model Accuracy: 98.03%")

st.write(
    """
The input to this network is a 28x28 pixel grayscale image, \
    which is flattened into a 1D vector of 784 elements \
        (28 * 28 = 784). This flattening is done in the \
            forward method using x.view(-1, 28*28).
         
We then used a hidden layer (nn.Linear) with 512 neurons. \
    Each of the 784 input values is connected to each of \
        the 512 neurons in this layer.The output of this \
            layer is then passed through a ReLU \
                (Rectified Linear Unit) activation function. \
                    ReLU graphs are just linear with a bend.
         
Lastly we have the output layer which takes in the 512 \
    neurons and outputs 1 of 10 (to represent the catogories \
        of number [0-9]. This is the classifcation step.)
"""
)

st.image("Pictures/Relu.png")
st.caption("The Relu function for reference.")

st.divider()
