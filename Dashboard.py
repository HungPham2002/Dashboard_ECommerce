import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
# import time

# Tạo Dashboard

# Using "with" notation
with st.sidebar:
    st.title('Customer Churn Prediction')
    st.header('Churn')
    st.write('A problem for B2C companies, churn is when a customer stop buying all products and services and becomes an ex-customer. Typically, this is a negative event for the company as it reduces the businesses revenue.')
    image_2 = Image.open('customerchurn.png')
    st.image(image_2)
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label="Temperature", value="31 °C", delta="1.0 °C",
        delta_color="inverse")
    with col2:
        st.metric(label="Wind", value="9 mph", delta="-8%",
        delta_color="inverse")
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label="Humidity", value="15%", delta="-4%",
        delta_color="inverse")
    with col2:
        st.metric(label="Location", value="HCM",
        delta_color="inverse")
    with st.container():
        st.subheader("Students:")
        st.text("Phạm Hữu Hùng - 20521371")
        st.text("Lê Văn Duy - 20521233")
        st.text("Nguyễn Phú Kiệt - 21522257")
        

st.title("Customer Churn Dashboard In 2022")
st.balloons()


# Load data
file = "telecom_customer_churn_datechurn.csv"
df = pd.read_csv(file)
# Biểu đồ 1
df['Churn Date'] = pd.to_datetime(df['Churn Date'])
monthly_churn = df.groupby(pd.Grouper(key='Churn Date', freq='M')).size() / len(df)

# Biểu đồ 2
big_cities = ['San Francisco', 'Los Angeles', 'San Diego', 'San Jose', 'Sacramento', 'Fresno', 'Long Beach', 'Oakland', 'Santa Ana', 'Anaheim', 'Riverside', 'Stockton']
df.loc[df['City'].isin(big_cities), 'city_group'] = 'Big cities'
medium_cities = ['Bakersfield', 'Chula Vista', 'Modesto', 'Irvine', 'Glendale', 'Fremont', 'San Bernardino', 'Fontana', 'Moreno Valley', 'Huntington Beach', 'Santa Clarita']
df.loc[df['City'].isin(medium_cities), 'city_group'] = 'Medium cities'
small_cities = ['Santa Rosa', 'Simi Valley', 'Merced', 'Napa', 'Redwood City', 'Watsonville', 'Madera', 'Grass Valley', 'Dixon', 'Corcoran', 'Susanville']
df.loc[df['City'].isin(small_cities), 'city_group'] = 'Small cities'



# Biểu đồ 4: Tỉ lệ churn theo độ tuổi
age_churn_count = df[df['Customer Status'] == 'Churned'].groupby('Age')['Customer ID'].count()
age_total_count = df.groupby('Age')['Customer ID'].count()
age_churn_rate = age_churn_count / age_total_count

# Vẽ biểu đồ 1 & 2 & 4:

with st.container():
    col1, col2, col3 = st.columns(3)
    with col1:
        fig1, ax1 = plt.subplots()
        ax1.plot(monthly_churn.index, monthly_churn.values)
        ax1.set_xlabel('Month')
        ax1.set_ylabel('Churn Rate')
        ax1.set_title('Monthly Churn Rate')
        st.pyplot(fig1)
    with col2:
        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='city_group', hue='Customer Status', ax=ax2)
        ax2.set_title('Number of Churned Customers by City Group')
        ax2.set_xlabel('City Group')
        ax2.set_ylabel('Number of Churned Customers')
        st.pyplot(fig2)
    with col3:
        plt.figure(figsize=(10,6))
        plt.scatter(age_churn_rate.index, age_churn_rate)
        plt.xlabel('Age')
        plt.ylabel('Churn Rate')
        plt.title('Churn Rate by Age')
        st.pyplot(plt)
        

# Biểu đồ 3
gender_churn_count = df[df['Customer Status'] == 'Churned'].groupby('Gender')['Customer ID'].count()
gender_total_count = df.groupby('Gender')['Customer ID'].count()
gender_churn_rate = gender_churn_count / gender_total_count
# Biểu đồ 6: Tỉ lệ khách hàng theo tình trạng khách hàng
status_count = df['Customer Status'].value_counts()
# Biểu đồ 7: Tỷ lệ churn theo từng danh mục churn
churn_by_category = df[df['Customer Status'] == 'Churned']['Churn Category'].value_counts()

# Vẽ Biểu đồ 3 & 6 & 7
# Tạo visual với kích thước và vị trí tùy chỉnh
with st.container():
    grid = st.container()
    with grid:
        col1, col2, col3 = st.columns(3)
        with col1:
            fig3, ax3 = plt.subplots()
            ax3.pie(gender_churn_rate, labels=gender_churn_rate.index, autopct='%1.1f%%')
            ax3.set_title('Churn Rate by Gender')
            st.pyplot(fig3)
        with col2:
            fig, ax = plt.subplots()
            ax.bar(churn_by_category.index, churn_by_category.values)
            ax.set_title('Churn by Churn Category')
            ax.set_xlabel('Churn Category')
            ax.set_ylabel('Number of Customers')
            st.pyplot(fig)
        with col3:
            fig, ax = plt.subplots()
            ax.pie(status_count, labels=status_count.index, autopct='%1.1f%%', startangle=90)
            ax.set_title('Customer Churn Rate')
            ax.axis('equal')
            st.pyplot(fig)
            
        
# Biểu đồ 5: Số lượng khách hàng theo từng loại hợp đồng và tình trạng khách hàng
contract_churn_count = df.groupby(['Contract', 'Customer Status'])['Customer ID'].count().unstack()
with st.container():
    fig, ax = plt.subplots()
    contract_churn_count.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Contract Type')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Customer Churn by Contract Type')
    ax.legend(title='Customer Status', loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)
# biểu đồ map
with st.container():
    random_matrix_1 = np.random.randn(100, 2) / [3, 1] + [36.782610 , -119.579324]
    random_matrix_2 = np.random.randn(100, 2) / [2, 1] + [39.782610 , -121.579324]
    random_matrix_3 = np.random.randn(100, 2) / [2, 1] + [37.782610 , -120.579324]
    # San Francisco
    random_matrix_4 = np.random.randn(100, 2) / [50, 50] + [37.76, -122.4]
    # Los Angeles
    random_matrix_5 = np.random.randn(100, 2) / [50, 50] + [34.0522342, -118.2436849]
    # San Diego
    random_matrix_6 = np.random.randn(100, 2) / [50, 50] + [32.715738, -117.1610838]
    random_matrix_7 = np.random.randn(100, 2) / [2, 1] + [35.0522342, -118.2436849]
    random_matrix_8 = np.random.randn(100, 2) / [1, 1] + [34.0522342, -116.2436849]
    df = pd.DataFrame(np.concatenate((random_matrix_1, random_matrix_2, random_matrix_3, random_matrix_4, random_matrix_5, random_matrix_6, random_matrix_7, random_matrix_8)),
                    columns=['lat', 'lon'])
    st.map(df)
    st.caption('Churn Distribution in California (2022)')
    
# Model Performance
with st.container():
    st.title("Model Prediction's Performance")