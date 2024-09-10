# Titanic Passenger Survival Analysis

## **Introduction**

This Streamlit application is an interactive exploration of the Titanic passenger survival dataset. It allows users to input passenger characteristics and predict their likelihood of survival based on a trained machine learning model. Additionally, the app visualizes survival patterns across different passenger classes.

## Running the Project

**1. Requirements**

* Python 3.12
* Streamlit (`pip install streamlit`)
* Pandas (`pip install pandas`)
* Pickle (`pip install pickle`) - for loading the pre-trained model
* Plotly (`pip install plotly`) - for interactive charts
* Numpy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)
* Seaborn (`pip install seaborn`) - Heatmaps
* Scikit-learn (`pip install scikit-learn`)

**2. Running with Docker**

(Assuming you have Docker installed)

* Build the Docker image:

```docker build -t titanic_survival_analysis .```

* Run the container:

```docker run -p 8501:8501 titanic_survival_analysis```

Open http://localhost:8501 in your web browser to access the Streamlit app.

**3. Running Locally**

* Clone this repository.

* Install the required libraries (pip install -r requirements.txt).

* Run the app using:

```streamlit run Titanic_app.py```

## Navigating the App

Once the app is running, you will see the left side of the screen will have a navigation bar.

![Sidebar Image](Pictures/Sidebar.png)

You will start on the "Titanic app" page which will display summary information on the data, features, model, and scaling functions used. 

![Home_One](Pictures/Home_1.png)

![Problem_statement](Pictures/Problem_statement.png)

![Model_Summ](Pictures/Model_Summ.png)

---

Navigating to the visualization page will display a searchable dataframe that will count the number of results from each search. Below that there will be various plots to represent the data in relation to survivability. 

![Viz_One](Pictures/Vis_1.png)

---

![Survival_Distro_by_Class](Pictures/Survival_Distro_by_Class.png)

This shows who survived or did not survived in terms of what class passenger they were. 

---

![Distro_by_sex](Pictures/Distro_by_sex.png)

Very few females did not survive but there was overall far more males on the Titanic.

---

![Age_Distro](Pictures/Age_Distro.png)

The majority of passengers were between 25 and 35 years old. 

---

![Heatmap](Pictures/Heatmap.png)

This heatmap displays teh correlations between features, visually. We want to focus on the bottom row for this problem so we can find what features best correspond to survival. 

---

![Survival](Pictures/Survival.png)

This is the overall survival rate of the Titanic Disaster. 

---

![Learning_Curve](Pictures/Learning_Curve.png)

This is the Learning curve which also displays the cross-validation score per training iteration.  

---

![Validation_Curve](Pictures/Validation_Curve.png)

This is the Validation curve showing a good fit for the data. 

---

![Kmeans++](Pictures/Kmeans++.png)

Here, we see an elbow plot displaying the optimal number of clusters for the Titanic Data based on KMeans++ Unsupervised Learning.  

---

![DBSCAN](Pictures/DBSCAN.png)

Here, we attempted to find trends using the DBSCAN unsupervised learning algorithm. We were unable to find any noticeable trends in the data.

---

![Advanced_1](Pictures/Advanced_1.png)

---

![Advanced_2](Pictures/Advanced_2.png)

---

The prediction tool has a few sliders that will allow you to input your own data for what class passenger you would be, your sex, your age, and approx. how much you would have spent on the fare to board the Titanic. This data will be entered in the model to give a prediction on whether or not you would have survived the Titanic. 

![Tool_One](Pictures/Tool_1.png)

---

![Dead](Pictures/Dead.png)

---

![Alive](Pictures/Alive.png)

---


## Conclusions

* First Class Passengers had a higher chance of survival.

* Women and children had a higher chance of survival.

* Survival chance decreased with age.

* The higher the fare that was paid, the better chance of survival. 
