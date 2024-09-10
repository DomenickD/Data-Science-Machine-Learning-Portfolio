"""the NLTK module explaination for NLP
for the streamlit app"""

import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer


# Download necessary NLTK resources (run this once)
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger")

st.header("NLTK to the Rescue!")
st.write(
    """
NLTK is like a professional organizer for your text:

* **Built-in Tools:**  It provides ready-to-use functions for each step.
* **Reliable:**  The functions are tested and optimized.
* **Consistent:**  Everyone gets the same results using NLTK.
"""
)

st.divider()

# Showcasing NLTK's Advantages (with Examples)
st.write("### Try it yourself!")
input_text_nltk = st.text_area("Enter your text for NLTK processing:", height=150)

st.divider()

if input_text_nltk:
    # Lowercasing
    st.subheader("Lowercasing")
    cleaned_text = input_text_nltk.lower()
    st.code("cleaned_text = input_text_nltk.lower()")
    st.write(cleaned_text)

    st.divider()

    # Tokenization
    st.subheader("Tokenization:")
    words = nltk.word_tokenize(cleaned_text)
    st.code("words = nltk.word_tokenize(cleaned_text)", language="python")
    st.write(words)

    st.divider()

    st.subheader("Part of Speech Tagging")
    pos_tagged_tokens = nltk.pos_tag(words)
    st.code("pos_tagged_tokens = nltk.pos_tag(words)", language="python")
    st.code(pos_tagged_tokens, language="text")
    # Dictionary mapping POS tags to definitions
    pos_tags = {
        "CC": "Coordinating conjunction (e.g., 'and', 'but')",
        "CD": "Cardinal number (e.g., 'one', 'two')",
        "DT": "Determiner (e.g., 'the', 'a', 'an')",
        "EX": "Existential there (e.g., 'there is')",
        "FW": "Foreign word",
        "IN": "Preposition or subordinating conjunction (e.g., 'in', 'of', 'for', 'because')",
        "JJ": "Adjective (e.g., 'big', 'happy')",
        "JJR": "Adjective, comparative (e.g., 'bigger', 'happier')",
        "JJS": "Adjective, superlative (e.g., 'biggest', 'happiest')",
        "MD": "Modal (e.g., 'can', 'could', 'will', 'would')",
        "NN": "Noun, singular or mass (e.g., 'dog', 'sugar')",
        "NNS": "Noun, plural (e.g., 'dogs', 'tables')",
        "NNP": "Proper noun, singular (e.g., 'John', 'London')",
        "NNPS": "Proper noun, plural (e.g., 'Smiths', 'Americans')",
        "PDT": "Predeterminer (e.g., 'all', 'both')",
        "POS": "Possessive ending (e.g., 's')",
        "PRP": "Personal pronoun (e.g., 'I', 'me', 'you', 'he', 'she', 'it')",
        "PRP$": "Possessive pronoun (e.g., 'my', 'your', 'his', 'her', 'its')",
        "RB": "Adverb (e.g., 'quickly', 'very')",
        "RBR": "Adverb, comparative (e.g., 'faster', 'more')",
        "RBS": "Adverb, superlative (e.g., 'fastest', 'most')",
        "RP": "Particle (e.g., 'up', 'off')",
        "TO": "'to'",
        "UH": "Interjection (e.g., 'oh', 'ah')",
        "VB": "Verb, base form (e.g., 'run', 'jump')",
        "VBD": "Verb, past tense (e.g., 'ran', 'jumped')",
        "VBG": "Verb, gerund or present participle (e.g., 'running', 'jumping')",
        "VBN": "Verb, past participle (e.g., 'run', 'jumped')",
        "VBP": "Verb, non-3rd person singular present (e.g., 'run', 'jump')",
        "VBZ": "Verb, 3rd person singular present (e.g., 'runs', 'jumps')",
        "WDT": "Wh-determiner (e.g., 'which', 'that')",
        "WP": "Wh-pronoun (e.g., 'who', 'what')",
        "WP$": "Possessive wh-pronoun (e.g., 'whose')",
        "WRB": "Wh-adverb (e.g., 'where', 'when')",
    }
    st.divider()
    st.subheader("Part of Speech Look Up")
    selected_tag = st.selectbox("Select a POS Tag:", list(pos_tags.keys()))

    if selected_tag:
        definition = pos_tags[selected_tag]
        st.write("**Definition:**", definition)

    st.divider()

    # Stopword Removal
    st.subheader("Stopword Removal:")
    stop_words = set(stopwords.words("english"))
    filtered_words = [
        word for word in words if word.lower() not in stop_words and word.isalnum()
    ]  # Also remove non-alphanumeric characters
    st.code(
        "filtered_words = [word for word in words \
            if word.lower() not in stopwords.words('english')]",
        language="python",
    )
    st.write(filtered_words)

    st.divider()

    # Lemmatization
    st.subheader("Lemmatization:")
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    st.code(
        """lemmatizer = WordNetLemmatizer()\nlemmatized_words = \
            [lemmatizer.lemmatize(word) for word in filtered_words]""",
        language="python",
    )
    st.write(lemmatized_words)

    st.divider()

    # Stemming
    st.subheader("Stemming:")
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    st.code(
        """stemmer = PorterStemmer()\nstemmed_words = \
            [stemmer.stem(word) for word in filtered_words]""",
        language="python",
    )
    st.write(stemmed_words)
