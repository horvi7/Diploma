import pandas as pd
from ydata_profiling import ProfileReport
import plotly.graph_objects as go
import numpy as np
import streamlit as st

def generate_profile_report(df, report_name):
    # profile = ProfileReport(df, title=report_name, explorative=True)
    # profile.to_file(report_name + ".html")

    year = None
    if "Year" in df.columns or "year" in df.columns:
        year_min = df["Year"].min() if "Year" in df.columns else df["year"].min()
        year_max = df["Year"].max() if "Year" in df.columns else df

        year = st.slider("Select Year", int(year_min), int(year_max), int(year_min))
        
    x_columns = [col for col in df.columns if col != "Year" and col != "year"]
    y_columns = [col for col in df.columns if col != "Year" and col != "year"]
    
    x_axis = st.selectbox("Select X-axis", x_columns)
    y_axis = st.selectbox("Select Y-axis", y_columns)

    
        



df = pd.read_csv("profiling_data/global_warming_dataset.csv")
generate_profile_report(df, "Global Warming Profile")