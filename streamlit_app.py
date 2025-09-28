# streamlit_app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px


# -------------------------------
# Load Data
# -------------------------------
dataset_path = "data/marketing_campaign_dataset.csv"  # Make sure CSV is in same folder
df = pd.read_csv(dataset_path)

# Clean numeric columns
df['Acquisition_Cost'] = df['Acquisition_Cost'].str.replace('[\$,]', '', regex=True).astype(float)
df['ROI'] = df['ROI'].astype(float)
df['Conversion_Rate'] = df['Conversion_Rate'].astype(float)
df['Clicks'] = df['Clicks'].astype(int)
df['Impressions'] = df['Impressions'].astype(int)
df['Engagement_Score'] = df['Engagement_Score'].astype(int)

# Calculated metrics
df['Engagement_per_Cost'] = df['Engagement_Score'] / df['Acquisition_Cost']
df['Conversions_per_Cost'] = (df['Conversion_Rate'] * df['Impressions']) / df['Acquisition_Cost']

df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.to_period('M')
df['Year'] = df['Date'].dt.year

# -------------------------------
# Sidebar Filters
# -------------------------------
st.sidebar.header("Filters")
companies = st.sidebar.multiselect("Select Companies", df['Company'].unique(), default=df['Company'].unique())
campaign_types = st.sidebar.multiselect("Select Campaign Types", df['Campaign_Type'].unique(), default=df['Campaign_Type'].unique())
years = st.sidebar.multiselect("Select Year(s)", df['Year'].unique(), default=df['Year'].unique())

# Apply filters
filtered_df = df[
    (df['Company'].isin(companies)) &
    (df['Campaign_Type'].isin(campaign_types)) &
    (df['Year'].isin(years))
]

# -------------------------------
# Dashboard Title
# -------------------------------
st.title("Marketing Campaign Budget Optimization Dashboard")

# -------------------------------
# KPI Metrics
# -------------------------------
total_budget = filtered_df['Acquisition_Cost'].sum()
avg_roi = filtered_df['ROI'].mean()
avg_engagement = filtered_df['Engagement_per_Cost'].mean()
avg_conversion_eff = filtered_df['Conversions_per_Cost'].mean()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Budget Spent", f"${total_budget:,.0f}")
col2.metric("Average ROI", f"{avg_roi:.2f}")
col3.metric("Avg Engagement per $", f"{avg_engagement:.2f}")
col4.metric("Conversions per $", f"{avg_conversion_eff:.2f}")

# -------------------------------
# Top 10 Campaigns by ROI
# -------------------------------
st.subheader("Top 10 Campaigns by ROI")
top10 = filtered_df.sort_values(by='ROI', ascending=False).head(10)
st.dataframe(top10[['Campaign_ID', 'Company', 'ROI', 'Acquisition_Cost', 'Engagement_per_Cost', 'Conversions_per_Cost']])

# -------------------------------
# ROI Bar Chart
# -------------------------------
st.subheader("ROI of Top 10 Campaigns")
plt.figure(figsize=(10,6))
sns.barplot(x='Campaign_ID', y='ROI', data=top10, palette="viridis")
plt.xticks(rotation=45)
plt.xlabel("Campaign ID")
plt.ylabel("ROI")
plt.tight_layout()
st.pyplot(plt)

# -------------------------------
# Engagement per Cost Chart
# -------------------------------
st.subheader("Engagement per $ Spent (Top 10 Campaigns)")
plt.figure(figsize=(10,6))
sns.barplot(x='Campaign_ID', y='Engagement_per_Cost', data=top10, palette="coolwarm")
plt.xticks(rotation=45)
plt.xlabel("Campaign ID")
plt.ylabel("Engagement per $")
plt.tight_layout()
st.pyplot(plt)

# -------------------------------
# Conversions per Cost Chart
# -------------------------------
st.subheader("Conversions per $ Spent (Top 10 Campaigns)")
plt.figure(figsize=(10,6))
sns.barplot(x='Campaign_ID', y='Conversions_per_Cost', data=top10, palette="magma")
plt.xticks(rotation=45)
plt.xlabel("Campaign ID")
plt.ylabel("Conversions per $")
plt.tight_layout()
st.pyplot(plt)



# -------------------------------
# Monthly ROI Trend
# -------------------------------
st.subheader("Monthly ROI Trend")
monthly_roi = filtered_df.groupby('Month').agg({'ROI':'mean'}).reset_index()
monthly_roi['Month'] = monthly_roi['Month'].astype(str)
fig_roi = px.line(monthly_roi, x='Month', y='ROI', markers=True, title="Average ROI Over Time")
fig_roi.update_layout(xaxis_title="Month", yaxis_title="Average ROI")
st.plotly_chart(fig_roi, use_container_width=True)

# -------------------------------
# Monthly Budget vs Engagement
# -------------------------------
st.subheader("Monthly Budget vs Engagement per $")
monthly_metrics = filtered_df.groupby('Month').agg({
    'Acquisition_Cost':'sum',
    'Engagement_per_Cost':'mean'
}).reset_index()
monthly_metrics['Month'] = monthly_metrics['Month'].astype(str)
fig_budget_engage = px.line(monthly_metrics, x='Month', y=['Acquisition_Cost', 'Engagement_per_Cost'],
                            markers=True, title="Monthly Budget vs Engagement per $")
fig_budget_engage.update_layout(xaxis_title="Month", yaxis_title="Values")
st.plotly_chart(fig_budget_engage, use_container_width=True)


# -------------------------------
# Download Filtered Data
# -------------------------------
st.download_button(
    label="Download Filtered Data",
    data=filtered_df.to_csv(index=False),
    file_name='filtered_campaign_data.csv',
    mime='text/csv'
)


