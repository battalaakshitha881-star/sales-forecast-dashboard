import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Sales Forecast Dashboard", layout="wide")
st.title("📦 Sales Forecast Dashboard")

uploaded_file = st.file_uploader("Upload your sales CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding='ISO-8859-1')
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[df['Quantity'] > 0]
    df = df[df['UnitPrice'] > 0]
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['Revenue'] = df['Quantity'] * df['UnitPrice']
    df['Month'] = df['InvoiceDate'].dt.to_period('M').astype(str)

    st.sidebar.header("Filters")
    country = st.sidebar.selectbox("Select Country", ["All"] + sorted(df['Country'].unique().tolist()))
    if country != "All":
        df = df[df['Country'] == country]

    total_revenue = df['Revenue'].sum()
    total_orders = df['InvoiceNo'].nunique()
    aov = total_revenue / total_orders

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"£{total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Avg Order Value", f"£{aov:,.2f}")

    monthly = df.groupby('Month')['Revenue'].sum().reset_index()
    monthly.columns = ['Month', 'Revenue']

    fig = px.line(monthly, x='Month', y='Revenue', title='Monthly Revenue Trend')
    st.plotly_chart(fig, use_container_width=True)

    monthly['MonthIndex'] = np.arange(len(monthly))
    X = monthly[['MonthIndex']]
    y = monthly['Revenue']
    model = LinearRegression().fit(X, y)

    future_idx = np.arange(len(monthly), len(monthly) + 3).reshape(-1, 1)
    forecast = model.predict(future_idx)

    st.subheader("📈 3-Month Revenue Forecast")
    forecast_df = pd.DataFrame({'Month': ['Next Month', 'Month+2', 'Month+3'], 'Forecasted Revenue': forecast})
    st.dataframe(forecast_df)
else:
    st.info("Please upload your online_retail CSV file to get started.")