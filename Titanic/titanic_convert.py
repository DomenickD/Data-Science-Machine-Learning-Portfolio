"""titanic ipny convert for lint"""

# Imports
import pandas as pd
import numpy as np

# Third-party library imports
from skopt import BayesSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    VotingClassifier,
)

# import models
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


# Load Data
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
gender_submission = pd.read_csv("gender_submission.csv")


# Data Cleaning
train["Sex_binary"] = train.Sex.map({"male": 0, "female": 1})
test["Sex_binary"] = test.Sex.map({"male": 0, "female": 1})

train["Age"] = train["Age"].fillna(round(train["Age"].mean()))
test["Age"] = test["Age"].fillna(round(test["Age"].mean()))
test["Fare"] = test["Fare"].fillna(round(test["Fare"].mean()))

columns_to_drop = ["PassengerId", "Name", "Sex", "Ticket", "Cabin", "Embarked"]

train = train.drop(columns_to_drop, axis=1)
test = test.drop(columns_to_drop, axis=1)


X_train = train[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_binary"]]
y_train = train["Survived"]
X_test = test[["Pclass", "Age", "SibSp", "Parch", "Fare", "Sex_binary"]]
y_test = gender_submission["Survived"]


# Create the scaler
scaler = MinMaxScaler()

# Create the logistic regression model
# lr_model = LogisticRegression(C=100, penalty='l2')
lr_model = LogisticRegression(
    C=0.046415888336127795, penalty="l2", class_weight=None, solver="saga"
)
# 'lr__C': 0.046415888336127795, 'lr__class_weight': None,
#  'lr__penalty': 'l2', 'lr__solver': 'saga'
nb_model = GaussianNB()
svc_model = SVC()
rfc_model = RandomForestClassifier()
dtc_model = DecisionTreeClassifier()

# Create the pipeline
pipeline_lr = Pipeline([("scaler", scaler), ("lr", lr_model)])

pipeline_nb = Pipeline([("scaler", scaler), ("nb", nb_model)])

pipeline_svc = Pipeline([("scaler", scaler), ("svc", svc_model)])

pipeline_rfc = Pipeline([("scaler", scaler), ("rfc", rfc_model)])

pipeline_dtc = Pipeline([("scaler", scaler), ("dtc", dtc_model)])


pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)
pipeline_lr_acc = accuracy_score(y_pred_lr, y_test)
print(f"lr accuracy = {pipeline_lr_acc}")

pipeline_nb.fit(X_train, y_train)
y_pred_nb = pipeline_nb.predict(X_test)
pipeline_nb_acc = accuracy_score(y_pred_nb, y_test)
print(f"nb accuracy = {pipeline_nb_acc}")

pipeline_svc.fit(X_train, y_train)
y_pred_svc = pipeline_svc.predict(X_test)
pipeline_svc_acc = accuracy_score(y_pred_svc, y_test)
print(f"svc accuracy = {pipeline_svc_acc}")

pipeline_rfc.fit(X_train, y_train)
y_pred_rfc = pipeline_rfc.predict(X_test)
pipeline_rfc_acc = accuracy_score(y_pred_rfc, y_test)
print(f"rfc accuracy = {pipeline_rfc_acc}")

pipeline_dtc.fit(X_train, y_train)
y_pred_dtc = pipeline_dtc.predict(X_test)
pipeline_dtc_acc = accuracy_score(y_pred_dtc, y_test)
print(f"dtc accuracy = {pipeline_dtc_acc}")


pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_train)
pipeline_lr_acc = accuracy_score(y_pred_lr, y_train)
print(f"lr accuracy ON TRAINING DATA= {pipeline_lr_acc}")


# Define parameter grids for each model
param_grid_lr = {
    "lr__penalty": ["l1", "l2"],
    # "lr__C": [0.001, 0.01, 0.1, 1, 10, 100, 110, 125, 150, 200],
    "lr__C": np.logspace(-3, 2, num=10),
    "lr__solver": ["liblinear", "saga"],
    "lr__class_weight": [None, "balanced"],
}

param_grid_nb = {"nb__var_smoothing": [1e-9, 1e-8, 1e-7]}

param_grid_svc = {"svc__C": [0.01, 0.1, 1, 10, 100]}

param_grid_rfc = {
    "rfc__n_estimators": [100, 200, 300],
    "rfc__max_features": [1, 2, 3, 4, 5],
}

param_grid_dtc = {
    "dtc__criterion": ["gini", "entropy"],
    "dtc__splitter": ["best", "random"],
}


# Set up BayesSearchCV for each pipeline
bayes_search_lr = BayesSearchCV(pipeline_lr, param_grid_lr, n_iter=10, cv=5, n_jobs=3)
bayes_search_nb = BayesSearchCV(pipeline_nb, param_grid_nb, n_iter=10, cv=5, n_jobs=3)
bayes_search_svc = BayesSearchCV(
    pipeline_svc, param_grid_svc, n_iter=10, cv=5, n_jobs=3
)
bayes_search_rfc = BayesSearchCV(
    pipeline_rfc, param_grid_rfc, n_iter=10, cv=5, n_jobs=3
)
bayes_search_dtc = BayesSearchCV(
    pipeline_dtc, param_grid_dtc, n_iter=10, cv=5, n_jobs=3
)

# Fit the models
bayes_search_lr.fit(X_train, y_train)
bayes_search_nb.fit(X_train, y_train)
bayes_search_svc.fit(X_train, y_train)
bayes_search_rfc.fit(X_train, y_train)
bayes_search_dtc.fit(X_train, y_train)


# Evaluate the models
print(f"Best parameters for Logistic Regression: {bayes_search_lr.best_params_}")
print(f"Best score for Logistic Regression: {bayes_search_lr.best_score_}")

print(f"Best parameters for Naive Bayes: {bayes_search_nb.best_params_}")
print(f"Best score for Naive Bayes: {bayes_search_nb.best_score_}")

print(f"Best parameters for SVC: {bayes_search_svc.best_params_}")
print(f"Best score for SVC: {bayes_search_svc.best_score_}")

print(f"Best parameters for Random Forest: {bayes_search_rfc.best_params_}")
print(f"Best score for Random Forest: {bayes_search_rfc.best_score_}")

print(f"Best parameters for Decision Tree: {bayes_search_dtc.best_params_}")
print(f"Best score for Decision Tree: {bayes_search_dtc.best_score_}")

# ---
#
# Best parameters for Logistic Regression: OrderedDict(
# {'lr__C': 0.1, 'lr__penalty': 'l2'})
#
# Best score for Logistic Regression: 0.7878601468834348
#
# ---
#
# Best parameters for Naive Bayes: OrderedDict(
# {'nb__var_smoothing': 1e-09})
#
# Best score for Naive Bayes: 0.7856631724311092
#
# ---
#
# Best parameters for SVC: OrderedDict({'svc__C': 100})
#
# Best score for SVC: 0.8193333751804659
#
# ---
#
# Best parameters for Random Forest: OrderedDict(
# {'rfc__max_features': 4, 'rfc__n_estimators': 200})
#
# Best score for Random Forest: 0.8114995919904588
#
# ---
#
# Best parameters for Decision Tree: OrderedDict(
# {'dtc__criterion': 'entropy', 'dtc__splitter': 'best'})
#
# Best score for Decision Tree: 0.7946582135459168
#
# ---
#

#


# Bagging
bagging_model = BaggingClassifier(
    estimator=pipeline_lr, n_estimators=10, random_state=42
)
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"lr_model: {accuracy}")


# Boosting
boosting_model = GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.1, random_state=42
)
boosting_model.fit(X_train, y_train)
y_pred = boosting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Boosting Model Accuracy: {accuracy:.2f}")


# Stacking
# We use multiple models so add the ones you wantto use in an
# array first - save the final estimator for the end
level1_models = [("svc", svc_model), ("nb", nb_model), ("rfc", rfc_model)]
# Define the final estimator (meta-learner) for the second level
final_estimator = lr_model  # for example. Can use anything else -


stacking_model = StackingClassifier(
    estimators=level1_models, final_estimator=final_estimator, cv=5
)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Stacking Model Accuracy: {accuracy:.2f}")


# Voting / Majority Method
voting_model = VotingClassifier(
    estimators=level1_models, voting="hard"
)  # Hard voting for classification - SOFT is regression
voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Majority Voting Model Accuracy: {accuracy:.2f}")
