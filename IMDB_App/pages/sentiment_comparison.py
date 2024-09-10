"""The Sentiment comparison for NLP streamlit app"""

import re
import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

st.header("NLTK (Manual) vs. LLM (LLama) Performance")
nltk.download("punkt")
nltk.download("stopwords")
st.divider()
st.subheader("Preprocessing")
# The SENTENCE for processing. If we change this,
# we need to ask the LLM so preprocess that one too
SENTENCE = "This is an example of how NLP works~! \
    Can it clean numbers? 1 2 3 %^&* :)"
# remove re
cleaned_text = re.sub(r"[^a-zA-Z\s]+", "", SENTENCE)
# lower
lower_SENTENCE = cleaned_text.lower()
# tokens
tokens = word_tokenize(lower_SENTENCE)
# stopword remove
stop_words = set(stopwords.words("english"))
filtered_tokens = [word for word in tokens if not word in stop_words]
# pos tagging
pos_word = nltk.pos_tag(filtered_tokens)
# stemming
stemmer = PorterStemmer()
stemmed_words = [stemmer.stem(word) for word in filtered_tokens]
# lemm
lemmatizer = WordNetLemmatizer()
lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]


st.write(
    f"""
The text for processing: {SENTENCE}

| NLP Step | NLTK | LLama (LLM) |
|---|---|---|
| Lower Case | {lower_SENTENCE} *(Also manually removed non-text characters)* \
    | this is an example of how nlp works~! can it clean numbers? 1 2 3 %^&* :) |
| Tokenization | {tokens} | [this, is, an, example, of, how, nlp, works~, can, \
    it, clean, numbers?, 1, 2, 3, %, ^%, &, *, :) ] |
| Stopwords | {filtered_tokens} | [example, how, works, it, clean] (removed: \
    is, an, of, nlp, numbers, this, can) |
| POS Tagging | {pos_word} | [this (DT), is (VBZ), an (DT), example (NN), of \
    (IN), how (WRB), nlp (NNP), works~! (.!), can (MD), it (PRP), clean (VB), \
        numbers? (?) 1 (CD) 2 (CD) 3 (CD) % (%), ^ (X), & (&), * (*), :) (:)] |
| Stemming | {stemmed_words} | [this, is, an, exampl, how, nlp, work, can, it, \
    clea, numb] (removed: example, of, nlp, works~!, clean, numbers?) |
| Lemmatization | {lemmatized_words} | [this, be, a, way, nlp, do, something, \
    can, you, make, clean, number] (removed: is, an, exampl, how, work, it, clea, numb) |
"""
)
st.divider()
st.subheader("Sentiment Prediction")

REVIEW_ONE = "How this character ever got hired by the post office is far \
    beyond me. The test that postal workers take is so difficult. \
        There is no way that a guy this stupid can work at the post \
            office. Everyone in this movie is just stupid and that is \
                probably the point of the movie. How they could go their \
                    entire lives and not see an elevator is also puzzling. \
                        I didn't take this movie too seriously but it \
                            was so stupid. Then he tries to start the \
                                car without his keys? Lots of horrible \
                                    scenes and horrible acting and this \
                                        movie is not funny at all. It's \
                                            just a sad stupid mess. I \
                                                liked the moms dress \
                                                    though.<br /><br />\
                                                        Send it back to \
                                                            sender as soon \
                                                                as possible."

REVIEW_TWO = "When this cartoon first aired I was under the impression that \
    it would be at least half way descent, boy was I wrong. I must admit \
        watching this cartoon is almost as painful as watching Batman and \
            Robin with George Clooney all those years ago. I watched a few \
                episodes and two of them had Batman literally get his ass \
                    kicked left and right by the Penguin who fought like Jet \
                        Li and beat the crap out of Batman and I watched \
                            another episode where Batman got his butt kicked \
                                again by the Joker, who apparently was using \
                                    Jackie Chan moves while flipping in the air \
                                        like a ninja. Since when were the Joker or \
                                            the Penguin ever a match for Batman ? \
                                                and worse yet when were Joker and \
                                                    Penguin Kung Fu counterparts of \
                                                        Jackie Chan and Jet Li. It's \
                              truly embarrassing, depressing and sad the \
                                  way the image of Batman is portrayed \
                                      in this show. The animation is awful \
                                          and the dialog is terrible. Being a \
                                            Batman fan since my boyhood I can \
                                                honestly and strongly advise \
                                                    you to stay away and avoid \
                                                        this show at all cost, \
                                                            because it doesn't \
                           project the true image of Batman. This cartoon is more like a \
                               wannabe Kung Fu Flick and if you really wanna see a classic \
                                   Batman cartoon I strongly recommend Batman the Animated \
                                       Series, but this cartoon is nothing more than a piece \
                                           of S---T! Get Batman: The Animates Series and don't \
                                               waste your time with this cartoon."

REVIEW_THREE = 'Good, funny, straightforward story, excellent Nicole Kidman \
    (I almost always like the movies she\'s in). This was a good "vehicle" for \
        someone adept at comedy and drama since there are elements of both. A \
            romantic comedy wrapped around two crime stories, great closing lines. \
                Chaplin, very good here, was also good in another good, but unpopular \
                    romantic comedy ("Truth about Cats & Dogs"). Maybe they\'re too \
            implausible. Ebert didn\'t even post a review for this. The great \
                "screwball" comedies obviously were totally implausible ("Bringing up \
                    Baby", etc.). If you\'ve seen one implausible comedy, you\'ve seen \
                        them all? Or maybe people are ready to move on from the 1930s. \
                            Weird. Birthday Girl is a movie I\'ve enjoyed several times. \
                                Nicole Kidman may be the "killer app" for home video.'

st.write(
    f"""
         
| Review Original | Model Prediction | LLama (LLM) Prediction | Actual Value |
|---|---|---|---|
| {REVIEW_ONE} | Negative | "overwhelmingly NEGATIVE" | 0 (negative) |
| {REVIEW_TWO} | Negative | "overwhelmingly NEGATIVE" | 0 (negative) |
| {REVIEW_THREE} | Positive | "POSITIVE" | 1 (positive) |

         
"""
)
st.caption(
    "Negative or 0 indicates a bad review sentiment. Positive or 1 indicates \
        a good review sentiment."
)

st.divider()

st.subheader("Summary Statistics")

st.write(
    """The sentiment analysis is a simple use case for NLP so we see \
        that the LLama LLM performed the same as our trained model \
            when it came down to a binary classifcation (prediction).

On 500 Samples from the 50,000 dataset.
         
Logistic Regression Accuracy: 0.85
         
LLM Accuracy: 0.9489795918367347
         
         
         
On 2,000 Samples from the 50,000 dataset.
         
Logistic Regression Accuracy: 0.8225
         
LLM Accuracy: 0.959785522788203
"""
)
