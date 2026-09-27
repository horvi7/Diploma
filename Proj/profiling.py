import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ydata_profiling import ProfileReport


def generate_profile_report(df, report_name):
    #YData Profiling
    profile = ProfileReport(df, title=report_name, explorative=True)
    profile.to_file(report_name + ".html")

    #Correlation Matrix
    st.title("Correlation Matrix")
    df.columns = [col.lower() for col in df.columns]

    df_numeric = df.select_dtypes(include=['number'])

    corr_x = df_numeric.columns
    corr_y = df_numeric.columns

    x_axis = st.multiselect("Select X-axis", corr_x)
    y_axis = st.multiselect("Select Y-axis", corr_y)

    
    #add correlation matrix with x_axis and y_axis
    st.title("Line Chart with Year filtered")
    if x_axis and y_axis:
        corr_matrix = df_numeric[x_axis + y_axis].corr()
    
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale='Viridis'))
        fig.update_layout(title="Correlation Matrix", xaxis_title="Features", yaxis_title="Features")
        st.plotly_chart(fig)
        

    #Line Chart with slider
    if "year" in df.columns:
        year_min = int(df["year"].min())
        year_max = int(df["year"].max())
        #Slider from min to max
        year = st.slider("Select Year", year_min, year_max)
        df = df[df["year"] == year]
    df_numeric = df.select_dtypes(include=['number'])
        
    x_columns = [col for col in df_numeric.columns]
    y_columns = [col for col in df_numeric.columns]
    
    x_axis = st.selectbox("Select X-axis", x_columns)
    y_axis = st.selectbox("Select Y-axis", y_columns)

    if x_axis and y_axis:
        df_grouped_avg = df.groupby(x_axis)[y_axis].mean().reset_index()
        df_grouped_median = df.groupby(x_axis)[y_axis].median().reset_index()
    fig = go.Figure()

    # Mean
    fig.add_trace(go.Scatter(
        x=df_grouped_avg[x_axis],
        y=df_grouped_avg[y_axis],
        mode='lines+markers',
        name='Mean'
    ))

    # Median
    fig.add_trace(go.Scatter(
        x=df_grouped_median[x_axis],
        y=df_grouped_median[y_axis],
        mode='lines+markers',
        name='Median'
    ))

    fig.update_layout(
        title=f"{y_axis} vs {x_axis}",
        xaxis_title=x_axis,
        yaxis_title=y_axis
    )

    st.plotly_chart(fig, width="content")
       


df = pd.read_csv("profiling_data/Data.csv")
generate_profile_report(df, "Global Warming Profile")