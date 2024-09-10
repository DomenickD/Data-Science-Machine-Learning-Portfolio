"""The streamlit page for teh process on NLP"""

import re
import streamlit as st
import spacy

st.header("The Natural Language Process")
st.divider()

st.subheader("Text Cleaning")

st.write(
    """
- Lowercasing: Convert all text to lowercase to ensure consistency and \
    avoid treating words like "Hello" and "hello" as different.

- Punctuation Removal: Remove punctuation marks (periods, commas, \
    exclamation points, etc.) as they often don't contribute much to the \
        meaning of the text in many NLP tasks.

- Number Removal: Decide whether numbers are relevant to your task. \
    If not, remove them.

- Special Character Removal: Get rid of any remaining special characters \
    (e.g., *, &, #) that don't add value.

- Whitespace Handling: Standardize whitespace by converting multiple spaces \
    into single spaces and removing leading or trailing spaces.

- Stopword Removal: Consider removing common words like "the," "a," "an," \
    etc. These words appear frequently but might not be significant for \
        tasks like text classification or topic modeling.
"""
)

st.divider()

st.write("### Try it yourself!")
input_text = st.text_area("Enter your text:", height=150)

st.divider()

if input_text:
    # Lowercasing
    CLEANED_TEXT = input_text.lower()
    st.subheader("Lowercased Text:")
    st.code(CLEANED_TEXT, language="text")

    st.divider()

    # Punctuation Removal
    CLEANED_TEXT = "".join(c for c in CLEANED_TEXT if c.isalnum() or c.isspace())
    st.subheader("Punctuation Removed:")
    st.code(CLEANED_TEXT, language="text")

    st.divider()

    # Number and Special Character Removal (Combined)
    CLEANED_TEXT = "".join(c for c in CLEANED_TEXT if c.isalpha() or c.isspace())
    st.subheader("Numbers & Special Characters Removed:")
    st.code(CLEANED_TEXT, language="text")

    st.divider()

    st.subheader("Manual Part of Speech Tagging")

    def simple_tagger(text):
        """function to tag text manually"""
        tagged = []
        for word in text.split():
            if word in ["the", "a", "an"]:
                tagged.append((word, "DT"))  # Determiner
            elif re.match(r"^[A-Za-z]+$", word):
                tagged.append((word, "NN"))  # Noun (assuming no verbs)
            else:
                tagged.append((word, "UNK"))  # Unknown
        return tagged

    st.code(
        """
    def simple_tagger(text):
        tagged = []
        for word in text.split():
            if word in ["the", "a", "an"]:
                tagged.append((word, "DT"))  # Determiner
            elif re.match(r"^[A-Za-z]+$", word):
                tagged.append((word, "NN"))  # Noun (assuming no verbs)
            else:
                tagged.append((word, "UNK"))  # Unknown
            return tagged
    """
    )

    tagged_tokens_manual = simple_tagger(CLEANED_TEXT)
    st.write(tagged_tokens_manual)

    stopwords = set(
        [
            "a",
            "an",
            "the",
            "and",
            "but",
            "or",
            "for",
            "nor",
            "as",
            "at",
            "by",
            "for",
            "from",
            "in",
            "into",
            "of",
            "on",
            "onto",
            "to",
            "with",
            "is",
            "am",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "i",
            "me",
            "my",
            "mine",
            "we",
            "us",
            "our",
            "ours",
            "you",
            "your",
            "yours",
            "he",
            "him",
            "his",
            "she",
            "her",
            "hers",
            "it",
            "its",
            "they",
            "them",
            "their",
            "theirs",
        ]
    )

    # Stopword Removal (manual)
    words = CLEANED_TEXT.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    st.subheader("Stopword Removal (Manual):")
    st.write(
        f"We have to define a list of stopwords before filtering through them:\n\n {stopwords}"
    )
    st.code(
        """filtered_words = [word for word in words if word.lower() not in stopwords]""",
        language="python",
    )
    st.write(filtered_words)

st.divider()
st.header("Tokenization")
st.write(
    """
**A "Token" is broadly defined as 3-4 characters.**         

- Word Tokenization: Break the text into individual words. \
    You can do this manually by splitting the text at whitespace \
        boundaries or using custom logic for handling punctuation.

- Sentence Tokenization: If needed, divide the text into separate \
    sentences. Look for punctuation marks like periods, question marks, \
        and exclamation points as potential sentence boundaries.
"""
)


st.divider()

if input_text:
    # Word Tokenization
    word_tokens = input_text.split()
    st.subheader("Word Tokens:")
    st.write(word_tokens)

    st.divider()

    # Sentence Tokenization (Simplified)
    sentences = input_text.split(".")  # Basic split on periods
    st.subheader("Sentence Tokens (Basic):")
    st.write(sentences)

st.divider()

st.header("Normalization")

st.write(
    """

- Stemming: Reduce words to their root form (e.g., "running," \
    "runs," "ran" become "run"). This can help group similar \
        words together.

- Lemmatization: A more sophisticated approach than stemming, \
    lemmatization reduces words to their base or dictionary form \
        (lemma) considering the part of speech. For example, \
            "better" becomes "good."
"""
)

st.code(
    """
# Simple stemming function (not as sophisticated as NLTK's)
def stem_word(word):
    # Basic rules for removing suffixes (you can add more)
    if word.endswith("ing"):
        return word[:-3]
    elif word.endswith("ed"):
        return word[:-2]
    elif word.endswith("s"):
        return word[:-1]
    return word

# Function for lemmatization (manual lemmatization is complex so \
# we had to use Spacy)
def lemmatize_word(word):
  #Lemmatizes a word using spaCy.
  doc = nlp(word)
  return doc[0].lemma_ 
""",
    language="python",
)


# Simple stemming function (not as sophisticated as NLTK's)
def stem_word(word):
    """function for word stemming"""
    # Basic rules for removing suffixes (you can add more)
    if word.endswith("ing"):
        return word[:-3]
    if word.endswith("ed"):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


# function for lemma
def lemmatize_word(word):
    """lem the word"""
    nlp = spacy.load("en_core_web_sm")
    # Lemmatizes a word using spaCy.
    doc = nlp(word)
    return doc[0].lemma_


if input_text:
    # Stemming (manual)
    stemmed_words = [stem_word(word) for word in filtered_words]
    st.subheader("Stemming (Manual):")
    st.code(
        """stemmed_words = [stem_word(word) for word in filtered_words]""",
        language="python",
    )
    st.write(stemmed_words)

    # Lemmatization (manual - placeholder)
    lemmatized_words = [lemmatize_word(word) for word in filtered_words]
    st.subheader("Lemmatization (Manual - With Spacy):")
    st.code(
        """lemmatized_words = [lemmatize_word(word) for word in filtered_words]""",
        language="python",
    )
    st.write(lemmatized_words)

st.divider()
