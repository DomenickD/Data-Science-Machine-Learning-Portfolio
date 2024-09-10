"""Intor to LDA for streamlit app NLP"""

import streamlit as st

st.header("What is LDA?")
st.write(
    """
Latent Dirichlet Allocation \
    (pronounced *lay-tent deer-ish-lay al-oh-kay-shun*) \
        is a method to uncover hidden themes or topics \
            in a collection of documents. \
                Think of it as a way to automatically \
                    group movie reviews based on similar subjects.
"""
)

st.divider()

st.subheader("Why Use LDA?")
st.write(
    """
LDA is useful when:
* You have a large collection of text data (like movie reviews).
* You want to understand the main themes or topics within the data.
* You don't know beforehand what those themes might be (LDA discovers them for you).
"""
)

st.divider()

# Section 2: LDA Code Example (Simplified)
st.header("LDA in Action (Simplified)")
st.code(
    """
# Import the LDA from sklearn 
from sklearn.decomposition import LatentDirichletAllocation
        
# Assign X to the reviews column of our Dataframe
X = count.fit_transform(df['review'].values)

# Train the LDA model
lda = LatentDirichletAllocation(n_components=10,
                                random_state=123,
                                learning_method='batch')

# Transform the data and fit the model
X_topics = lda.fit_transform(X)

# Get the top 5 words for each topic
n_top_words = 5
feature_names = count.get_feature_names_out()

for topic_idx, topic in enumerate(lda.components_):
    print(f'Topic {(topic_idx + 1)}:')
    print(' '.join([feature_names[i]
        for i in topic.argsort()
            [:-n_top_words - 1:-1]]))
"""
)
st.write(
    """
OUTPUT:


Topic 1:

worst minutes awful script stupid

Topic 2:

family mother father children girl

Topic 3:

american war dvd music tv

Topic 4:

human audience cinema art sense

Topic 5:

police guy car dead murder

Topic 6:

horror house sex girl woman

Topic 7:

role performance comedy actor performances

Topic 8:

series episode war episodes tv

Topic 9:

book version original read novel

Topic 10:

action fight guy guys cool
"""
)

st.write(
    "This code snippet demonstrates the basic steps \
        involved in training an LDA model and extracting topics."
)

st.divider()

# Section 3: Interpreting LDA Results
st.header("Interpreting LDA Results")
st.write(
    """
LDA produces topics as lists of words.  \
    You'd then interpret these word lists to \
        understand what each topic represents.  \
            For example, a topic might look like this:
"""
)
st.code(
    """(0, '0.020*"acting" + 0.015*"performance" + 0.012*"role"\
          + 0.010*"oscar" + ...')"""
)
st.write("This could represent a topic about acting and awards.")
