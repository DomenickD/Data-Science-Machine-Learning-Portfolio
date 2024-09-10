"""Plots page for streamlit app"""
import streamlit as st
import plotly.express as px  # pip install plotly
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, learning_curve, validation_curve

import Titanic.Titanic_App.titanic_streamlit_app as titanic


titanic_data = pd.read_csv("titanic_data.csv")

st.header("Visualizing the Data")
st.divider()
st.subheader("The Combined Titanic Dataframe")
titanic_data = pd.read_csv("titanic_data.csv")
columns_to_drop = [
    "Unnamed: 0",
    "Pclass_1",
    "Pclass_2",
    "Pclass_3",
    "Sex_binary",
]  # , "Name", "Ticket", "Cabin", "Sex", "Embarked"
titanic_data = titanic_data.drop(columns_to_drop, axis=1)

column_to_filter_by = st.selectbox("Choose a column to filter by", titanic_data.columns)
filter_options = st.multiselect(
    "Filter by", options=titanic_data[column_to_filter_by].unique()
)

# Filtering data based on selection
if filter_options:
    filtered_data = titanic_data[titanic_data[column_to_filter_by].isin(filter_options)]
else:
    filtered_data = titanic_data

st.dataframe(filtered_data)
st.write(f"{filtered_data["PassengerId"].count()} results are displayed.")
st.caption(
    "This is the combined dataset for the Titanic Data. \
        It included the Train and Test csv files from Kaggle."
)

st.write("""---""")


st.subheader("Survival Distribution by Passenger Class")

counts_class_1 = titanic_data[titanic_data["Pclass"] == 1]["Survived"].value_counts()
counts_class_2 = titanic_data[titanic_data["Pclass"] == 2]["Survived"].value_counts()
counts_class_3 = titanic_data[titanic_data["Pclass"] == 3]["Survived"].value_counts()

# Correctly order counts (If 'Survived' and 'Not Survived' aren't present, fill with 0)
counts_class_1 = counts_class_1.reindex([1, 0], fill_value=0)
counts_class_2 = counts_class_2.reindex([1, 0], fill_value=0)
counts_class_3 = counts_class_3.reindex([1, 0], fill_value=0)

# Create the graph
fig = go.Figure(
    data=[
        go.Bar(
            name="Class 1",
            x=["Survived", "Not Survived"],
            y=counts_class_1,
            marker_color="gold",
        ),
        go.Bar(
            name="Class 2",
            x=["Survived", "Not Survived"],
            y=counts_class_2,
            marker_color="skyblue",
        ),
        go.Bar(
            name="Class 3",
            x=["Survived", "Not Survived"],
            y=counts_class_3,
            marker_color="grey",
        ),
    ]
)
fig.update_layout(barmode="group", xaxis_range=[-0.5, 1.5], bargap=0.1)
st.plotly_chart(fig)

st.caption("""Survival status of the passenger (0 = No, 1 = Yes).""")

st.write("""---""")
st.subheader("Survival Distribution by Sex")

fig = go.Figure(
    data=[
        go.Bar(
            name="Female",
            x=["Survived", "Not Survived"],
            y=titanic_data[titanic_data["Sex"] == "female"]["Survived"].value_counts(),
            marker_color="pink",
        ),
        go.Bar(
            name="Male",
            x=["Survived", "Not Survived"],
            y=titanic_data[titanic_data["Sex"] == "male"]["Survived"].value_counts(),
            marker_color="teal",
        ),
    ]
)
fig.update_layout(barmode="group", xaxis_range=[-0.5, 1.5], bargap=0.1)
st.plotly_chart(fig)

st.caption(
    """This shows a side by side of the amount of males to \
    females who survived the Titanic and did not survive."""
)

st.write("""---""")

st.subheader("Age Distribution (Grouped)")
fig = px.histogram(
    titanic_data, x="Age", nbins=9, range_x=[0, 80]
)  # Adjust range as needed
fig.update_layout(
    xaxis_title="Age (Groups of 10)",
    bargap=0.05,
    xaxis_range=[-5, 85],
    yaxis_range=[0, 600],
)
fig.update_traces(marker_color="lightgreen")
st.plotly_chart(fig)


st.caption("This plot represents the distribution of ages onboard the Titanic.")

st.write("""---""")
st.subheader("Titanic Heatmap")

# move "Survived" to teh far right and bottom.
temp_df = titanic_data["Survived"]
titanic_data = titanic_data.drop(["Survived"], axis=1)
titanic_data = titanic_data.merge(
    temp_df, how="left", left_index=True, right_index=True
)

titanic_numbers = titanic_data.select_dtypes(include=np.number)
# mask = np.zeros_like(titanic_numbers.corr())
# mask[np.triu_indices_from(mask)] = True

plt.figure(figsize=(8, 6))
# sns.heatmap(titanic_numbers.corr(), annot=True, cmap='coolwarm', mask=mask)
sns.heatmap(titanic_numbers.corr(), annot=True, cmap="coolwarm")
plt.xticks(rotation=45)
plt.yticks(rotation=45)
st.pyplot(plt.gcf())

st.caption("This Heatmap shows the correlation between features and 'Survived'.")

st.divider()
st.subheader("Survival Distribution")
survived_counts = titanic_data["Survived"].value_counts().sort_index()
fig = px.pie(
    survived_counts,
    values=survived_counts.values,
    names=["Did Not Survive", "Survived"],
    color_discrete_sequence=["red", "green"],  # Optional: Adjust colors
)

st.plotly_chart(fig)

st.caption(
    """**This plot represents the distribution of survivors on board.**     
- Red represents those who did NOT survive. 
- Green represents those who survived."""
)

st.write("""---""")

st.subheader("Learning Curve")
train_sizes, train_scores, test_scores = learning_curve(
    titanic.pipeline,
    titanic.X_train,
    titanic.y_train,
    cv=10,
    n_jobs=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
)

# Calculate means and standard deviations for plotting
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

# Plot the learning curve
fig, ax = plt.subplots()
ax.plot(train_sizes, train_mean, label="Training score")
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
ax.plot(train_sizes, test_mean, label="Cross-validation score")
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)

ax.set_xlabel("Training examples")
ax.set_ylabel("Score")
ax.legend()
st.pyplot(fig)

st.caption("This is the learning curve for the model.")


st.divider()

st.subheader("Validation Curve: Varying C")

PARAM_NAME = "lr__C"  # Choose a hyperparameter to vary
param_range = np.logspace(-3, 2, num=5)

train_scores, test_scores = validation_curve(
    titanic.pipeline,
    titanic.X_train,
    titanic.y_train,
    param_name=PARAM_NAME,
    param_range=param_range,
    cv=10,
)

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

cv_scores = cross_val_score(titanic.pipeline, titanic.X_train, titanic.y_train, cv=10)
cv_mean = np.mean(cv_scores)
cv_std = np.std(cv_scores)

st.caption(f"Accuracy Score: {titanic.accuracy:.4f} ± {cv_std:.4f}")

st.divider()
st.subheader("Unsupervised Learning - K-Means++")
st.image(r"Pictures\KMEANS++_Elbow.png")
st.caption("Here is the attempt at applying Kmeans++ to the Titanic Dataset.")

st.divider()
st.subheader("Unsupervised Learning - DBSCAN")
st.image(r"Pictures\DBSCAN_DEFAULT.png")
st.caption(
    "Here is the attempt at applying DBSCAN to the Titanic Dataset. No noticable trends."
)
