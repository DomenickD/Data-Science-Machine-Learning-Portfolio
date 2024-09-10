"""Entry point for streamlit splash page"""

import pickle
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.model_selection import train_test_split as tts
from sklearn import metrics
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import xgboost as xgb


columns = [
    "Gr Liv Area",
    "Total Bsmt SF",
    "Full Bath",
    "TotRms AbvGrd",
    "Fireplaces",
    "Lot Area",
    "Overall Qual",
    "SalePrice",
]

df = pd.read_csv("AmesHousing.txt", sep="\t", usecols=columns)

# df["Central_Air_Binary"] = df['Central Air'].map({'N': 0, 'Y': 1})
# df = df.drop("Central Air", axis = 1)

df = df.dropna()
Y = df.SalePrice
X = df.drop("SalePrice", axis=1)


pipeline = Pipeline([("scaler", MinMaxScaler()), ("xgb", xgb.XGBRegressor())])
X_train, X_test, y_train, y_test = tts(X, Y, test_size=0.2, random_state=42)

pipeline.fit(X_train, y_train)
# for the plot on next page
selector = SelectKBest(score_func=f_regression, k=6)
X_train_new = selector.fit_transform(X_train, y_train)
best_feature_indices = selector.get_support(indices=True)
original_column_names = X.columns.to_list()  # Store the names
best_features = np.array(original_column_names)[best_feature_indices]

# for corrlation matrix plot on next page
df_corr = pd.concat([X, Y], axis=1, ignore_index=False)
corr_matrix = df_corr.corr()

st.header("House Prices Predictor")
st.write(
    """
**Data Source**:  \
    [Github](https://github.com/rasbt/machine-learning-book/blob/main/ch09/AmesHousing.txt)       

**Author**: Domenick Dobbs          
"""
)
st.divider()
st.image("Pictures/Ames_Downtown.png")
st.caption("Downtown Ames, Iowa in the summer.")
st.caption("Source: https://www.worldatlas.com/cities/ames-iowa.html")
st.divider()
st.write(
    """
## Data Background

The Ames Housing Dataset was compiled by Dean De Cock \
    (Iowa State University) in 2011 for use in research and education.
The data captures information on residential home sales \
    in Ames, Iowa between 2006 and 2010.
The Full dataset contains 2930 records and it is a commonly \
    used dataset for Exploratory Data Analysis for Machine Learning Regression.     
    
---
         
## Goal 
         
The primary goal of this project is to build a predictive model that \
    can reliably estimate the sale price of a house in Ames, Iowa. \
        This model will leverage various housing attributes, \
            like living area, number of bedrooms, and overall \
                quality, to uncover patterns and make informed \
                    predictions.
         
---

"""
)

st.write(
    """
## Columns Used in Analysis

| Column Name | Data Type | Description |
|---|---|---|
| Lot Area | Continuous | Lot size (sq. ft.) |
| Overall Qual | Ordinal | Rates overall material and finish (1-10)| 
| Total Bsmt SF | Continuous | Total basement area (sq. ft.) |
| Gr Liv Area | Continuous | Above-grade living area (sq. ft.) |
| Full Bath | Discrete | Full bathrooms above grade |
| TotRms AbvGrd | Discrete | Total rooms above grade (excluding bathrooms) |
| Fireplaces | Discrete | Number of fireplaces |
| SalePrice | Continuous | Sale price ($) |

***
"""
)

y_pred = pipeline.predict(X_test)
mse = metrics.mean_squared_error(y_test, y_pred)
mae = metrics.mean_absolute_error(y_test, y_pred)
r2 = metrics.r2_score(y_test, y_pred)


param_dist = {
    "xgb__n_estimators": randint(100, 400),
    "xgb__max_depth": randint(2, 8),
    "xgb__learning_rate": uniform(0.01, 0.2),
    "xgb__subsample": uniform(0.6, 0.4),
    "xgb__colsample_bytree": uniform(0.6, 0.4),
    "xgb__reg_alpha": uniform(0, 1),
}

# @st.cache_resource
# def get_fitted_model(_pipeline, X_train, y_train, _param_dist):
#     random_search = RandomizedSearchCV(_pipeline, _param_dist,
# n_iter=100, cv=5, scoring='neg_mean_squared_error', verbose=1)
#     random_search.fit(X_train, y_train)  # Fit the model once
#     return random_search.best_estimator_

# Usage on Main Page
# best_model = get_fitted_model(pipeline, X_train, y_train, param_dist)

best_model = Pipeline(
    steps=[
        ("scaler", MinMaxScaler()),
        (
            "xgb",
            xgb.XGBRegressor(
                base_score=None,
                booster=None,
                callbacks=None,
                colsample_bylevel=None,
                colsample_bynode=None,
                colsample_bytree=0.8196235298606114,
                device=None,
                early_stopping_rounds=None,
                enable_categorical=False,
                eval_metric=None,
                feature_types=None,
                gamma=None,
                grow_policy=None,
                importance_type=None,
                interaction_constraints=None,
                learning_rate=0.03186751598813972,
                max_bin=None,
                max_cat_threshold=None,
                max_cat_to_onehot=None,
                max_delta_step=None,
                max_depth=4,
                max_leaves=None,
                min_child_weight=None,
                monotone_constraints=None,
                multi_strategy=None,
                n_estimators=268,
                n_jobs=None,
                num_parallel_tree=None,
                random_state=None,
            ),
        ),
    ]
)

best_model.fit(X_train, y_train)
predictions = best_model.predict(X_test)
mse_post = metrics.mean_squared_error(y_test, predictions)
mae_post = metrics.mean_absolute_error(y_test, predictions)
r2_post = metrics.r2_score(y_test, predictions)
# print(f"Model: XGB Regression\nMSE: {mse_post:.2f}\nMAE:
# {mae_post:.2f}\nR-squared: {r2_post:.2f}\n-----------------")

st.write(
    f"""
## Model Summary
- **Model Type**: I'm using an XGBoost Regressor model. This is a powerful \
    type of gradient boosting algorithm that builds decision trees in an \
        ensemble to make predictions. It's known for its accuracy and \
            ability to handle a wide variety of data types.

- **Feature Scaling**: I've applied a MinMaxScaler to the data. This \
    scaling technique helps ensure that all features in the dataset \
        have a similar range (typically between 0 and 1), which can \
            improve the performance of the model.
 
---
         
##  Model Performance Metrics before Hyperparameter Tuning
- Mean Squared Error: {mse:.2f}
- Mean Absolute Error : {mae:.2f}
- R-Squared: {r2:.4f}

##  Model Performance Metrics AFTER Hyperparameter Tuning
- Mean Squared Error: {mse_post:.2f}
- Mean Absolute Error : {mae_post:.2f}
- R-Squared: {r2_post:.4f}

 """
)
st.divider()

FILENAME = "xgb_pipeline_minmaxscaler.pkl"
with open(FILENAME, "wb") as f:
    pickle.dump(pipeline, f)
