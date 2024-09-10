"""Entry point for streamlit app and splash page"""

import streamlit as st

st.header("The IMDB Dataset for Natural Language Processing")
st.caption("By: Domenick Dobbs")

st.divider()

st.subheader("About the data")
st.write(
    """
The IMDB dataset is a collection of 50,000 movie reviews from the \
Internet Movie Database (IMDB) website. The dataset is balanced, \
with 25,000 positive and 25,000 negative reviews. 
         
The primary use of the IMDB dataset is to train and evaluate models \
that can determine the sentiment expressed in a piece of text (e.g., \
movie review, product review). The binary labels make it suitable for \
supervised learning tasks.
         
The dataset consists of a Review Text column and a Sentiment Label \
column.
         
Some challenges Challenges and Considerations here include:

- Sarcasm and Subtlety: Some reviews may contain sarcasm or express \
sentiment in subtle ways, making accurate classification challenging.
         
- Data Bias: The dataset may contain biases inherent in the original \
reviews, which could affect model performance.
         
- Pre-processing: Raw reviews often require preprocessing steps like \
tokenization, stop-word removal, and potentially stemming or \
lemmatization before being used in models.
         
"""
)

st.divider()

st.subheader("Word Cloud of all the text")
st.image("Pictures/wordcloud_all.png", width=800)
st.caption("The larger the word, the more frequently it appears.")
st.divider()
st.subheader("Honorable Mentions from the data")
st.write(
    """
df["review"][610][:300]

I was having just about the worst day of my life. Then I stumbled on this \
cute film, watched it, and now I'm ready to go out & kiss a streetlamp.\
<br /><br />I have to admit, I only watched it for 2 reasons: \
VERA-ELLEN'S LEGS. But it's really so much more.
         
df["review"][607][:300]
         
When this cartoon first aired I was under the impression that it would \
be at least half way descent, boy was I wrong. I must admit watching this \
cartoon is almost as painful as watching Batman and Robin with George \
Clooney all those years ago.
"""
)
