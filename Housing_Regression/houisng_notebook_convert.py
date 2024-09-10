"""Formerly an ipynb converted to py for linting and documentation"""

import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    BaggingRegressor,
    StackingRegressor,
    VotingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Lasso, ElasticNet, LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

import xgboost as xgb


columns_predifined = [
    "Overall Qual",
    "Overall Cond",
    "Gr Liv Area",
    "Central Air",
    "Total Bsmt SF",
    "SalePrice",
]

df = pd.read_csv("AmesHousing.txt", sep="\t", usecols=columns_predifined)


df["Central_Air_Binary"] = df["Central Air"].map({"N": 0, "Y": 1})
df = df.dropna(axis=0)
y = df["SalePrice"]
X = df.drop(["Central Air", "SalePrice"], axis=1)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


xgb_model = xgb.XGBRegressor()

rfr_model = RandomForestRegressor(
    max_depth=10, min_samples_leaf=4, min_samples_split=10, n_estimators=300
)
svr_model = SVR(C=10, epsilon=0.2)
lass_model = Lasso(alpha=10)
eln_model = ElasticNet(alpha=0.01, l1_ratio=0.95)
lr_model = LinearRegression()
# scaler
scaler = MinMaxScaler()

# Create the pipeline
pipeline_xgb = Pipeline([("scaler", scaler), ("xgb", xgb_model)])

pipeline_rfr = Pipeline([("scaler", scaler), ("rfr", rfr_model)])


pipeline_svr = Pipeline([("scaler", scaler), ("svr", svr_model)])

pipeline_lass = Pipeline([("scaler", scaler), ("lass", lass_model)])

pipeline_eln = Pipeline([("scaler", scaler), ("eln", eln_model)])

pipeline_lr = Pipeline([("scaler", scaler), ("lr", lr_model)])


pipeline_xgb.fit(X_train, y_train)
pipeline_rfr.fit(X_train, y_train)
pipeline_svr.fit(X_train, y_train)
pipeline_lass.fit(X_train, y_train)
pipeline_eln.fit(X_train, y_train)
pipeline_lr.fit(X_train, y_train)

ypred_xgb = pipeline_xgb.predict(X_test)
ypred_rfr = pipeline_rfr.predict(X_test)
ypred_svr = pipeline_svr.predict(X_test)
ypred_lass = pipeline_lass.predict(X_test)
ypred_eln = pipeline_eln.predict(X_test)
ypred_lr = pipeline_lr.predict(X_test)

r2_xgb = r2_score(y_test, ypred_xgb)
r2_rfr = r2_score(y_test, ypred_rfr)
r2_svr = r2_score(y_test, ypred_svr)
r2_lass = r2_score(y_test, ypred_lass)
r2_eln = r2_score(y_test, ypred_eln)
r2_lr = r2_score(y_test, ypred_lr)

print(f"r2_xgb: {r2_xgb}")
print(f"r2_rfr: {r2_rfr}")
print(f"r2_svr: {r2_svr}")
print(f"r2_lass: {r2_lass}")
print(f"r2_eln: {r2_eln}")
print(f"r2_lr: {r2_lr}")

# XGBoost Regressor
xgb_params = {
    "xgb__n_estimators": [100, 200, 300],  # Number of boosting rounds
    "xgb__max_depth": [3, 5, 7],  # Maximum depth of a tree
    "xgb__learning_rate": [
        0.1,
        0.01,
        0.001,
    ],  # Step size shrinkage used in update to prevents overfitting
    "xgb__subsample": [0.8, 0.9, 1.0],  # Subsample ratio of the training instance
    "xgb__colsample_bytree": [
        0.8,
        0.9,
        1.0,
    ],  # Subsample ratio of columns when constructing each tree
    "xgb__gamma": [
        0,
        0.25,
        0.5,
    ],  # Minimum loss reduction required to make a further partition on a leaf node of the tree
}

# Random Forest Regressor
rfr_params = {
    "rfr__n_estimators": [100, 200, 300],  # Number of trees in the forest
    "rfr__max_depth": [None, 10, 20],  # Maximum depth of the tree
    "rfr__min_samples_split": [
        2,
        5,
        10,
    ],  # Minimum number of samples required to split an internal node
    "rfr__min_samples_leaf": [
        1,
        2,
        4,
    ],  # Minimum number of samples required to be at a leaf node
}

# Support Vector Regressor
svr_params = {
    "svr__kernel": ["linear", "rbf"],  # Kernel type
    "svr__C": [
        0.1,
        1,
        10,
    ],  # The strength of the regularization is inversely proportional to C
    "svr__gamma": [
        "scale",
        "auto",
    ],  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
}

# Lasso Regressor
lasso_params = {
    "lass__alpha": [0.01, 0.1, 1, 10],  # Constant that multiplies the L1 term.
}

# Elastic Net Regressor
elasticnet_params = {
    "eln__alpha": [0.01, 0.1, 1, 10],  # Constant that multiplies the penalty terms
    "eln__l1_ratio": [
        0.1,
        0.5,
        0.7,
        0.9,
        0.95,
        0.99,
        1,
    ],  # The ElasticNet mixing parameter
}


random_search_xgb = RandomizedSearchCV(
    pipeline_xgb, param_distributions=xgb_params, cv=5, scoring="r2", n_jobs=3
)
random_search_rfr = RandomizedSearchCV(
    pipeline_rfr, param_distributions=rfr_params, cv=5, scoring="r2", n_jobs=3
)
random_search_svr = RandomizedSearchCV(
    pipeline_svr, param_distributions=svr_params, cv=5, scoring="r2", n_jobs=3
)
random_search_lass = RandomizedSearchCV(
    pipeline_lass, param_distributions=lasso_params, cv=5, scoring="r2", n_jobs=3
)
random_search_eln = RandomizedSearchCV(
    pipeline_eln, param_distributions=elasticnet_params, cv=5, scoring="r2", n_jobs=3
)

random_search_xgb.fit(X_train, y_train)
random_search_rfr.fit(X_train, y_train)
random_search_svr.fit(X_train, y_train)
random_search_lass.fit(X_train, y_train)
random_search_eln.fit(X_train, y_train)

best_xgb = random_search_xgb.best_estimator_
best_rfr = random_search_rfr.best_estimator_
best_svr = random_search_svr.best_estimator_
best_lass = random_search_lass.best_estimator_
best_eln = random_search_eln.best_estimator_

print(f"best_xgb: {best_xgb}")
print(f"best_rfr: {best_rfr}")
print(f"best_svr: {best_svr}")
print(f"best_lass: {best_lass}")
print(f"best_eln: {best_eln}")


# Bagging
bagging_model = BaggingRegressor(estimator=xgb_model, n_estimators=10, random_state=42)
bagging_model.fit(X_train, y_train)
y_pred = bagging_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"xgb_model: {r2}")


# Boosting
booster = GradientBoostingRegressor().fit(X_train, y_train)
y_pred = booster.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"booster_model: {r2}")


# Stacking
level1_models = [("svr", svr_model), ("lass", lass_model), ("rfr", rfr_model)]
# Define the final estimator (meta-learner) for the second level
final_estimator = xgb_model

stacking_model = StackingRegressor(
    estimators=level1_models, final_estimator=final_estimator, cv=5
)
stacking_model.fit(X_train, y_train)
y_pred = stacking_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Stacking Model R2: {r2:.2f}")


# voting
voting_model = VotingRegressor(estimators=level1_models)
voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

print(f"Majority Voting Model R2: {r2:.2f}")
