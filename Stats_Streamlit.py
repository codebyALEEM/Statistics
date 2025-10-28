# Import Labraries
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

# set up the title and discription 
st.title('Sales Data Analysis for Retail Store')
st.write('Analyze Sales data for various product categories.')

# Generate synthetic sales data
def generate_data():
    np.random.seed(42)
    data={
        'product_id': range(1,21),
        'product_name': [f'Product{i}' for i in range(1,21)],
        'category' : np.random.choice(['Electronic','Clothing','Home','Sports'],20),
        'units_sold' : np.random.poisson(lam=20,size=20),
        'sale_date' : pd.date_range(start='2023-01-01',periods=20,freq='D')        
    }
    return pd.DataFrame(data)

sales_data = generate_data()

#Display the sales data
st.subheader('Sales_Data')
st.dataframe(sales_data)

#Descriptive Statestics
st.subheader("Descriptive Statistics")
descriptive_stats =sales_data['units_sold'].describe()
st.write(descriptive_stats)

mean_sales = sales_data['units_sold'].mean()
median_sales = sales_data['units_sold'].median()
mode_sales = sales_data['units_sold'].mode()

st.write(f"Mean Units Sold:{mean_sales}")
st.write(f"Mdian Units Sold:{median_sales}")
st.write(f"Mode Units Sold:{mode_sales}")

#Group statistic by category
category_stats =sales_data.groupby('category')['units_sold'].agg(['sum','mean','std']).reset_index()
category_stats.columns=['Category','Total Units Sold','Average Units Sold','Std Dev of Units Sold']
st.subheader('Category Statistics')
st.dataframe(category_stats)

#Inferential Statistics
confidence_level = 0.95
degree_freedom = len(sales_data['units_sold'])-1
sample_mean = mean_sales
sample_standard_error = sales_data['units_sold'].std()/np.sqrt(len(sales_data['units_sold']))

# t-score for the confidence level
t_score = stats.t.ppf((1+confidence_level)/2,degree_freedom)
margin_of_error = t_score * sample_standard_error
confidence_interval = (sample_mean-margin_of_error,sample_mean+margin_of_error)

st.subheader("Confidence Interval for Mean Units Sold")
st.write(confidence_interval)

#Hypothesis Testing
t_statistic,p_value = stats.ttest_1samp(sales_data['units_sold'],20)

st.subheader('Hypothesis Testing (t-test)')
st.write(f"T-statistic : {t_statistic},P-value:{p_value}")

if p_value < 0.05:
    st.write('Reject the null hypothesis : The mean units sold is significantly different from 20.')
else:
    st.write('Fail to reject the null hypothesis : The mean units sold is not significantly different from 20.')
    
# Visualization
st.subheader('Visualizaions')

#Histogram of units sold
plt.figure(figsize=(10,6))
sns.histplot(sales_data['units_sold'],bins=10,kde=True,color='purple')
plt.title('Distribution of Units sold')
plt.xlabel('Units sold')
plt.ylabel('Frequency')
plt.axvline(mean_sales,color='red',linestyle='--',label='Mean')
plt.axvline(median_sales,color='blue',linestyle='--',label='Median')
plt.axvline(mode_sales.iloc[0], color='green', linestyle='--', label='Mode')
plt.legend()
st.pyplot(plt)

#Box plot for Units sold by category
plt.figure(figsize=(10,6))
sns.boxplot(x='category',y='units_sold',data=sales_data,color='teal')
plt.title('Boxplot of units sold by category')
plt.xlabel('Category')
plt.ylabel('Total Units Sold')
st.pyplot(plt)

# Bar plot for total units sold by category
plt.figure(figsize=(10,6))
sns.barplot(x='Category',y='Total Units Sold',data=category_stats,color='brown')
plt.title('Total Units Sold by Category')
plt.xlabel('Category')
plt.ylabel('Total Units Sold')
st.pyplot(plt)

