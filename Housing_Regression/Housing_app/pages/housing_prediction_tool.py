"""The prediction tool v1 for housing"""

import pickle
import streamlit as st
import pandas as pd

with open("xgb_pipeline_minmaxscaler.pkl", "rb") as f:
    loaded_pipeline = pickle.load(f)
st.write(
    """
## Housing Price Exploration Tool

---
         
Let's explore factors that might influence housing prices. 
"""
)

lot_area = st.slider("Lot Area (sq. ft.):", 0, 20000, 5000)
overall_qual = st.select_slider(
    "Overall Quality (1-10):", options=range(1, 11), value=5
)
total_bsmt_sf = st.slider("Total Basement SF (sq. ft.):", 0, 3000, 1000)
gr_liv_area = st.slider("Above Grade Living Area (sq. ft):", 0, 5000, 1500)
full_bath = st.number_input("Full Bathrooms:", min_value=0, max_value=5, value=2)
TotRms_AbvGrd = st.number_input(
    "Total rooms above grade:", min_value=0, max_value=5, value=2
)
Fireplaces = st.number_input("Number of fireplaces:", min_value=0, max_value=5, value=2)

# columns = ['Gr Liv Area', 'Total Bsmt SF', 'Full Bath',
# 'TotRms AbvGrd', 'Fireplaces', 'Lot Area', 'Overall Qual', 'SalePrice']

user_input = pd.DataFrame(
    {
        "Lot Area": [lot_area],
        "Overall Qual": [overall_qual],
        "Total Bsmt SF": [total_bsmt_sf],
        "Gr Liv Area": [gr_liv_area],
        "Full Bath": [full_bath],
        "TotRms AbvGrd": [TotRms_AbvGrd],
        "Fireplaces": [Fireplaces],
    }
)

if st.button("Predict"):
    # user_input_scaled = Scalerminmax.transform(user_input)
    prediction = loaded_pipeline.predict(user_input)[0]

    st.subheader("Predicted Sale Price")
    st.write(f"${prediction:,.2f}")
