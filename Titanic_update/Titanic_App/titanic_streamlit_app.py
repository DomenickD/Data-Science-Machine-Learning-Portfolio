"""streamlit entrypoint app"""

import pickle
import os.path
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score

from sklearn.impute import SimpleImputer


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


st.set_page_config(page_title="Titanic Analysis", page_icon="👋")

st.write(
    """
# Titanic Classification Dataset
**Data Source:** [Kaggle](https://www.kaggle.com/c/titanic/data)
     
**Author:** Domenick Dobbs"""
)
st.divider()

st.image("Pictures/Titanic.png")
st.caption("Source: https://cdn.britannica.com/79/4679-050-BC127236/Titanic.jpg")
st.divider()

st.write(
    """
## Overview of this project from Kaggle
The sinking of the Titanic is one of the most infamous shipwrecks in history.

On April 15, 1912, during her maiden voyage, the widely considered \
    “unsinkable” RMS Titanic sank after colliding with an iceberg. \
        Unfortunately, there weren't enough lifeboats for everyone \
            onboard, resulting in the death of 1502 out of 2224 passengers and crew.

While there was some element of luck involved in surviving, it seems \
    some groups of people were more likely to survive than others.

In this challenge, we ask you to build a predictive model that answers \
    the question: “what sorts of people were more likely to survive?” \
        using passenger data (ie name, age, gender, socio-economic class, etc).
"""
)

st.write(
    """
---
## Disclaimer about the data
This dataset does not include all of the PEOPLE from the actual Titanic. 
There are 1309 rows of data for *passengers* in this Kaggle Dataset. \
    There were 2240 total Passengers **and** Crew.
As a result, the 931 crew members are not accounted for. 
"""
)
st.write(
    """
---
## Problem Statement
The goal of this project is to develop a predictive model that accurately \
    identifies factors influencing passenger survival rates \
            during the tragic sinking of the RMS Titanic. 
         By analyzing historical passenger data, we seek to uncover patterns \
            and relationships between individual characteristics 
         (such as age, gender, socio-economic class, cabin location, etc.) \
            and their likelihood of survival.
         """
)

st.write(
    """
---       
## List of Column Names and what the values represent
         
| Column Name    | Description                                                                 |
|----------------|-----------------------------------------------------------------|
| PassengerId    | A unique numerical identifier assigned to each passenger.                   |
| Survived       | Survival status of the passenger (0 = No, 1 = Yes).                         |
| Pclass         | The passenger's ticket class (1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class). |
| Name           | The passenger's full name.                                                  |
| Sex            | The passenger's gender (male, female).                                      |
| Age            | The passenger's age in years. Fractional values may exist for \
    young children.|
| SibSp          | The number of siblings or spouses traveling with the passenger.             |
| Parch          | The number of parents or children traveling with the passenger.             |
| Ticket         | The passenger's ticket number.                                              |
| Fare           | The price the passenger paid for their ticket.                              |
| Cabin          | The passenger's cabin number (if recorded).                                 |
| Embarked       | The passenger's port of embarkation (C = Cherbourg, Q = Queenstown, \
    S = Southampton). |
---
"""
)

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")

# train = pd.concat([train, pd.get_dummies(train["Pclass"], prefix='Pclass')], axis=1)
# test = pd.concat([test, pd.get_dummies(test["Pclass"], prefix='Pclass')], axis=1)

train["Sex_binary"] = train.Sex.map({"male": 0, "female": 1})
test["Sex_binary"] = test.Sex.map({"male": 0, "female": 1})


class RoundingTransformer(BaseEstimator, TransformerMixin):
    """copy and paste"""

    def fit(self):
        """Can delete this"""
        return self

    def transform(self, x_value):
        """transform method"""
        x_value = x_value.round()
        return x_value


imputer = SimpleImputer(strategy="mean")
rounder = RoundingTransformer()
preprocessing_pipeline = Pipeline([("imputer", imputer), ("rounder", rounder)])

# Create the scaler
min_max_scaler = MinMaxScaler()

# Create the logistic regression model
model = LogisticRegression()

# Create the pipeline
pipeline = Pipeline(
    [
        ("preprocessing", preprocessing_pipeline),
        ("scaler", min_max_scaler),
        ("lr", model),
    ]
)
# train['Age'].fillna(value = round(train['Age'].mean()), inplace = True)
# test['Age'].fillna(value = round(test['Age'].mean()), inplace = True)
# test['Fare'].fillna(value = round(test['Fare'].mean()), inplace = True)


test_merged = pd.merge(test, gender_submission, how="inner")
titanic_data = pd.concat([train, test_merged], ignore_index=True)

if not os.path.exists("titanic_data.csv"):
    titanic_data.to_csv("titanic_data.csv")

# Assign Features and Labels
X_train = train[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_binary"]]
# X_train = train[["Pclass", "Age", "Fare", "Sex_binary"]]
y_train = train["Survived"]
X_test = test[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_binary"]]
# X_test = test[["Pclass", "Age", "Fare", "Sex_binary"]]
y_test = gender_submission["Survived"]

# train_features = train[["Age", "Sex_binary", "Pclass", "Fare"]]
# train_labels = train["Survived"]
# test_features = test[["Age", "Sex_binary", "Pclass", "Fare"]]
# test_labels = gender_submission["Survived"]
# train_features_scaled = min_max_scaler.fit_transform(train_features)
# test_features_scaled = min_max_scaler.fit_transform(test_features)
# model.fit(train_features_scaled, train_labels)
# y_predict = model.predict(test_features_scaled)
# accuracy = accuracy_score(test_labels, y_predict)

# # Fit the pipeline to your training data
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

param_grid = {
    "lr__penalty": ["l1", "l2"],
    # "lr__C": [0.001, 0.01, 0.1, 1, 10, 100, 110, 125, 150, 200],
    "lr__C": np.logspace(-3, 2, num=10),
    "lr__solver": ["liblinear", "saga"],
    "lr__class_weight": [None, "balanced"],
}

# GridSearchCV
# grid_search = GridSearchCV(pipeline, param_grid, cv=10)
# grid_search.fit(X_train, y_train)

best_params = {
    "C": 0.5994842503189409,
    "class_weight": None,
    "penalty": "l1",
    "solver": "liblinear",
}

pipeline = Pipeline(
    [
        ("preprocessing", preprocessing_pipeline),
        ("scaler", MinMaxScaler()),
        ("lr", LogisticRegression(**best_params)),
    ]
)

# # Fit the pipeline to your training data
pipeline.fit(X_train, y_train)

# Make predictions
predictions = pipeline.predict(X_test)

accuracy_post = accuracy_score(y_test, predictions)

with open("my_model.pkl", "wb") as f:
    pickle.dump(model, f)


st.write(
    f"""
## Model Details
         
This data was run against multiple models and multiple normalization methods. 
The highest ratings were from the logistic regression model \
    with a standardized MinMaxScalar provided by Sci-kit learn.
         
Model Accuracy before Hyperparameter Tuning: 
         
**{round((100*accuracy), 2)}%**.

Model Accuracy AFTER Hyperparameter Tuning: 

**{round((100*accuracy_post), 2)}**%.

"""
)
st.divider()


with open("scaler.pkl", "wb") as f:
    pickle.dump(min_max_scaler, f)
