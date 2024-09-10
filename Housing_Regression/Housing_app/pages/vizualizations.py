"""Viz page for streamlit app for housing"""

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.model_selection import learning_curve, validation_curve, cross_val_score
from ..housing__streamlit_app import (
    pipeline,
    X_train,
    y_train,
    best_model,
    corr_matrix,
    df,
    best_features,
    selector,
    r2,
)

st.header("Visualizations")

st.divider()

column_to_filter_by = st.selectbox("Choose a column to filter by", df.columns)
filter_options = st.multiselect("Filter by", options=df[column_to_filter_by].unique())

# Filtering data based on selection
if filter_options:
    filtered_data = df[df[column_to_filter_by].isin(filter_options)]
else:
    filtered_data = df

st.dataframe(filtered_data)
st.write(f"{filtered_data["Lot Area"].count()} results are displayed.")

st.divider()


st.subheader("Feature Importance")
fig = px.bar(
    x=best_features,
    y=selector.scores_[:6],
    labels={"x": "Features", "y": "F-Regression Score"},
)
fig.update_layout(
    xaxis={"tickangle": 30},
    xaxis_range=[-0.5, 5.5],  # Adjust tick label rotation
    yaxis_range=[0, 4500],  # Adjust tick label rotation
    bargap=0.2,
)
st.plotly_chart(fig)
st.caption("""Using an algorthim, we were able to determine the top 6 features.""")
st.divider()

###
st.subheader("Distribution of Sale Prices")
fig = px.histogram(
    df, x="SalePrice", color_discrete_sequence=px.colors.qualitative.Light24_r
)
fig.update_layout(xaxis_title="Sale Price ($)", yaxis_title="Count")
st.plotly_chart(fig)
st.caption(
    "This helps us visually understand the overall shape of sale price distribution."
)
st.divider()
###

###
st.subheader("Relationship between Above Grade Living Area and Sale Price")
fig = px.scatter(
    df,
    x="Gr Liv Area",
    y="SalePrice",
    trendline="ols",
    color="Overall Qual",
    color_discrete_map={"1": "blue", "5": "orange", "10": "green"},
)
fig.update_layout(
    xaxis_title="Above Grade Living Area (sq.ft.)", yaxis_title="Sale Price ($)"
)
st.plotly_chart(fig)
st.caption(
    "This helps to visualize if there's a positive correlation (and whether it's linear or not)."
)
st.divider()
###
###
st.subheader("Sale Price Box Plots by Overall Quality")
fig = px.box(
    df,
    x="Overall Qual",
    y="SalePrice",
    color_discrete_sequence=px.colors.qualitative.Light24,
)
fig.update_layout(xaxis_title="Overall Quality", yaxis_title="Sale Price ($)")
st.plotly_chart(fig)
st.caption(
    "See how the distribution and median sale prices differ across \
        quality ratings. This shows that some the highest quality \
            homes can be less than $200k."
)
st.divider()
###

st.subheader("Correlation Heatmap")
# mask = np.triu(np.ones_like(housing.corr_matrix, dtype=bool))
# Create the interactive heatmap (with masking)
fig, ax = plt.subplots(figsize=(10, 8))
# sns.heatmap(housing.corr_matrix, annot=True, cmap='coolwarm', ax=ax, mask=mask)
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
plt.xticks(rotation=45)
plt.yticks(rotation=45)
st.pyplot(fig)
st.caption(
    """This Heatmap will display correlations between features. \
        We are only concerned with what correlates with the SalesPrice feature."""
)

st.divider()
st.subheader("Learning Curve for XGB Regressor")

train_sizes, train_scores, test_scores = learning_curve(
    best_model, X_train, y_train, cv=5, n_jobs=3, train_sizes=np.linspace(0.1, 1.0, 10)
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

fig, ax = plt.subplots()

ax.plot(train_sizes, train_mean, "o-", color="r", label="Training score")
ax.fill_between(
    train_sizes, train_mean - train_std, train_mean + train_std, color="r", alpha=0.1
)
ax.plot(train_sizes, test_mean, "o-", color="g", label="Cross-validation score")
ax.fill_between(
    train_sizes, test_mean - test_std, test_mean + test_std, color="g", alpha=0.1
)


ax.set_xlabel("Training examples")
ax.set_ylabel("Score")
ax.legend(loc="best")
ax.grid(True)
st.pyplot(fig)

st.caption("High Variance.")

st.divider()
st.subheader("Validation Curve: Varying C")

PARAM_NAME = "xgb__learning_rate"  # Choose a hyperparameter to vary
param_range = np.logspace(-3, 2, num=5)

train_scores, test_scores = validation_curve(
    pipeline, X_train, y_train, param_name=PARAM_NAME, param_range=param_range, cv=10
)

# Calculate means and standard deviations
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the validation curve
fig, ax = plt.subplots()
ax.plot(param_range, train_mean, label="Training score")
ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
ax.plot(param_range, test_mean, label="Cross-validation score")
ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1)

ax.set_xlabel("C (Regularization parameter)")
ax.set_ylabel("Score")
ax.legend()
ax.set_xscale("log")
st.pyplot(fig)

cv_scores = cross_val_score(pipeline, X_train, y_train, cv=10)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

st.caption(f"R2 Score: {r2:.4f} ± {cv_std:.4f}")

st.divider()
st.subheader("Unsupervised Learning - K-Means++")
st.image(r"Pictures\K-Means++_Elbow_Plot.png")
st.caption("Here is the attempt at applying Kmeans++ to the Ames Housing Dataset.")

st.divider()
st.subheader("Unsupervised Learning - DBSCAN")
st.image(r"Pictures\DBSCAN_Default_Params.png")
st.caption("Here is the attempt at applying DBSCAN to the Ames Housing Dataset.")
