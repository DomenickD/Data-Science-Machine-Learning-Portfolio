"""The Mnist overview for the streamlit app"""

import streamlit as st

st.header("Neural Networks with MNIST")

st.write("By: Domenick Dobbs")

st.divider()

st.subheader("Introduction to the Dataset")

st.write(
    """The MNIST dataset is a collection of 70,000 handwritten digits \
        (0-9) that are used to train and test machine 
         learning algorithms. The dataset is divided into 60,000 \
            training images and 10,000 test images. Each image is a 28x28
         pixel grayscale image."""
)

st.divider()

st.subheader("The MNIST Dataset at a glance")
st.image(r"Pictures/MNIST_full.png")
st.caption(
    "This sample shows the pattern of what type of data is in our \
        dataset. Please note it is already cleaned and gray scale. \
            This image is from: \
                https://towardsdatascience.com/solve-the-mnist-image-classification-problem-9a2865bcf52a"
)

st.divider()

st.subheader("The partial MNIST Dataset")
st.image(r"Pictures/MNIST_part.png")
st.caption(
    "This is a picture of a smaller sample of \
        the dataset so it is easier to see from: \
        https://datasets.activeloop.ai/docs/ml/datasets/mnist/"
)

st.divider()

st.subheader("MNIST Header")
st.image(r"Pictures\Sample_MNIST.png")
st.caption("This is a picture of the first 5 images in the dataset we loaded to use.")

st.divider()

st.subheader("Pytorch Framework")

st.write(
    """
**PyTorch**, developed by Facebook's AI Research lab (FAIR), is a popular \
    open-source deep learning framework known for its dynamic computation \
        graph and user-friendly interface. It was first released in 2016 \
            and has quickly gained traction among researchers and \
                developers due to its flexibility and ease of use. \
                    PyTorch excels in rapid prototyping and experimentation, \
                        making it ideal for research-oriented projects. 
         
Its Pythonic nature and intuitive API allow for seamless integration \
    with the wider Python ecosystem, making it a favorite among developers. \
        Unlike TensorFlow's static graph approach, PyTorch's dynamic graph \
        enables more efficient debugging and model modification. \
            PyTorch also boasts strong community support and a \
                wealth of tutorials and resources, making it a \
                    great choice for beginners and experienced \
                        practitioners alike."""
)

st.write(
    "We will be using Pytorch to approach the MNIST Dataset and \
        classify the images into numbers using a Neural Network."
)

st.divider()

st.subheader("Tensorflow Framework")

st.write(
    """
**TensorFlow**, developed by Google Brain, is a powerful open-source \
    deep learning framework known for its comprehensive ecosystem and \
        robust production capabilities. Released in 2015, \
            TensorFlow has become a cornerstone in the deep learning \
                community due to its scalability and deployment options. \
                    It excels in large-scale projects and industrial \
                        applications, offering optimized performance \
                            and support for distributed training. 

TensorFlow's static graph approach, while requiring more upfront \
    planning, allows for efficient model optimization and deployment \
        in diverse environments. While initially considered less \
            beginner-friendly than PyTorch, TensorFlow 2.0 introduced Eager \
                Execution, making it more accessible and allowing for dynamic \
                    graph building. TensorFlow also boasts a vast collection of \
                        pre-trained models and specialized libraries \
                            (e.g., TensorFlow Lite for mobile and \
                                embedded devices), making it a \
                                    versatile tool for a wide range \
                                        of deep learning tasks.
"""
)
