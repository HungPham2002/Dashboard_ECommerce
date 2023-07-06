import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_curve, auc, roc_auc_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# import time

# create a Dashboard

# Using "with" notation
with st.sidebar:
    st.title('Customer Churn Prediction')
    st.header('Churn')
    st.write('A problem for B2C companies, churn is when a customer stop buying all products and services and becomes an ex-customer. Typically, this is a negative event for the company as it reduces the businesses revenue.')
    image_2 = Image.open('customerchurn.png')
    st.image(image_2)
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label="Temperature", value="30 °C", delta="1.0 °C",
        delta_color="inverse")
    with col2:
        st.metric(label="Wind", value="15 km/h", delta="-8%",
        delta_color="inverse")
    col1,col2 = st.columns(2)
    with col1:
        st.metric(label="Humidity", value="73%", delta="-4%",
        delta_color="inverse")
    with col2:
        st.metric(label="Location", value="HCM",
        delta_color="inverse")
    with st.container():
        st.subheader("Students:")
        st.text("Phạm Hữu Hùng - 20521371")
        st.text("Lê Văn Duy - 20521233")
        

st.title("Customer Churn Dashboard In 2022")
st.balloons()


# Load data
file = "telecom_customer_churn_datechurn.csv"
df = pd.read_csv(file)
# Diagram 1
df['Churn Date'] = pd.to_datetime(df['Churn Date'])
monthly_churn = df.groupby(pd.Grouper(key='Churn Date', freq='M')).size() / len(df)

# Diagram 2
big_cities = ['San Francisco', 'Los Angeles', 'San Diego', 'San Jose', 'Sacramento', 'Fresno', 'Long Beach', 'Oakland', 'Santa Ana', 'Anaheim', 'Riverside', 'Stockton']
df.loc[df['City'].isin(big_cities), 'city_group'] = 'Big cities'
medium_cities = ['Bakersfield', 'Chula Vista', 'Modesto', 'Irvine', 'Glendale', 'Fremont', 'San Bernardino', 'Fontana', 'Moreno Valley', 'Huntington Beach', 'Santa Clarita']
df.loc[df['City'].isin(medium_cities), 'city_group'] = 'Medium cities'
small_cities = ['Santa Rosa', 'Simi Valley', 'Merced', 'Napa', 'Redwood City', 'Watsonville', 'Madera', 'Grass Valley', 'Dixon', 'Corcoran', 'Susanville']
df.loc[df['City'].isin(small_cities), 'city_group'] = 'Small cities'



# Diagram 4: Prop churn by age
age_churn_count = df[df['Customer Status'] == 'Churned'].groupby('Age')['Customer ID'].count()
age_total_count = df.groupby('Age')['Customer ID'].count()
age_churn_rate = age_churn_count / age_total_count

# Draw Diagram 1 & 2 & 4:

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
        

# Diagram 3
gender_churn_count = df[df['Customer Status'] == 'Churned'].groupby('Gender')['Customer ID'].count()
gender_total_count = df.groupby('Gender')['Customer ID'].count()
gender_churn_rate = gender_churn_count / gender_total_count
# Biểu đồ 6: Prop customer status
status_count = df['Customer Status'].value_counts()
# Diagram 7: Prop churn by churn category
churn_by_category = df[df['Customer Status'] == 'Churned']['Churn Category'].value_counts()

# Draw Diagram 3 & 6 & 7
# create visual with size and position custom
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
            
        
# Diagram 5: the number of customer displayed by contracts & customer status
contract_churn_count = df.groupby(['Contract', 'Customer Status'])['Customer ID'].count().unstack()
with st.container():
    fig, ax = plt.subplots()
    contract_churn_count.plot(kind='bar', stacked=True, ax=ax)
    ax.set_xlabel('Contract Type')
    ax.set_ylabel('Number of Customers')
    ax.set_title('Customer Churn by Contract Type')
    ax.legend(title='Customer Status', loc='center left', bbox_to_anchor=(1, 0.5))
    st.pyplot(fig)
#  map (Just random, LOL)
with st.container():
    random_matrix_1 = np.random.randn(200, 2) / [3, 1] + [36.782610 , -119.579324]
    random_matrix_2 = np.random.randn(200, 2) / [2, 1] + [39.782610 , -121.579324]
    random_matrix_3 = np.random.randn(200, 2) / [2, 1] + [37.782610 , -120.579324]
    # San Francisco
    random_matrix_4 = np.random.randn(200, 2) / [50, 50] + [37.76, -122.4]
    # Los Angeles
    random_matrix_5 = np.random.randn(200, 2) / [50, 50] + [34.0522342, -118.2436849]
    # San Diego
    random_matrix_6 = np.random.randn(200, 2) / [50, 50] + [32.715738, -117.1610838]
    random_matrix_7 = np.random.randn(200, 2) / [2, 1] + [35.0522342, -118.2436849]
    random_matrix_8 = np.random.randn(200, 2) / [1, 1] + [34.0522342, -116.2436849]
    df = pd.DataFrame(np.concatenate((random_matrix_1, random_matrix_2, random_matrix_3, random_matrix_4, random_matrix_5, random_matrix_6, random_matrix_7, random_matrix_8)),
                    columns=['lat', 'lon'])
    st.map(df)
    st.caption('Churn Distribution in California (2022)')
    
# Model Performance
with st.container():
    st.title("Model Performances")

# Load data
data = pd.read_csv('mycsvfile.csv')

# Split data
X = data.drop(columns="customerstatus")
y = data["customerstatus"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4, stratify=y)

# Feature scaling
col = ['totalcharges', 'avgmonthlylongdistancecharges', 'monthlycharge', 'totalrevenue', 'totallongdistancecharges',
       'tenureinmonths', 'totallongdistancecharges', 'totalextradatacharges']
scaler = StandardScaler()
X_train[col] = StandardScaler().fit_transform(X_train[col])
X_test[col] = StandardScaler().fit_transform(X_test[col])

tab1, tab2, tab3, tab4, tab5 = st.tabs(["KNN", "Logistic Regression", "SVM", "Decision Tree", "Random Forest", ])
with tab1:
    # 1. KNN MODEL
    # Train model
    algorithm = 'brute'
    metric = 'euclidean'
    n_neighbors = 20
    good_modelknn = KNeighborsClassifier(algorithm=algorithm, metric=metric, n_neighbors=n_neighbors)
    good_modelknn.fit(X_train, y_train)

    # Test model
    predknn = good_modelknn.predict(X_test)
    accknn = accuracy_score(y_test, predknn)
    precknn = precision_score(y_test, predknn)
    recaknn = recall_score(y_test, predknn)

    # Classification report
    classification_report_data = classification_report(y_test, predknn, output_dict=True)
    classification_report_dfKnn = pd.DataFrame(classification_report_data).transpose()

    # Confusion matrix
    confusion_matrix_df = pd.DataFrame(confusion_matrix(y_test, predknn))

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, good_modelknn.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='KNeighborsClassifier (AUC = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")

    # Display on Streamlit
    st.write("## KNN Model Performance")
    col1, col2 = st.columns([1,2])
    with col1:
        st.write("### Confusion Matrix")
        st.dataframe(confusion_matrix_df)
        st.write("### Evaluate Model")
        st.write(f"Test accuracy = {accknn: .4f}")
        st.write(f"Test precision = {precknn: .4f}")
        st.write(f"Test recall = {recaknn: .4f}")
        st.write("### Classification Report")
        st.dataframe(classification_report_dfKnn)
    with col2:
        st.write("### ROC Curve")
        st.pyplot(plt)

with tab2:
    #2. Logistic Regression
    # Train model
    C = 1000
    max_iter = 1000
    good_modelL = LogisticRegression(C = C, max_iter=max_iter) # create model 
    good_modelL.fit(X_train,y_train) # train model

    # Evaluate model
    predL = good_modelL.predict(X_test)
    accL = accuracy_score(y_test, predL)
    precL = precision_score(y_test, predL)
    recaL = recall_score(y_test, predL)
    
    fprL, tprL, thresholdsL = roc_curve(y_test, good_modelL.predict_proba(X_test)[:, 1])
    roc_aucL = auc(fprL, tprL)
    
    # Classification Report

    classification_report_data = classification_report(y_test, predL, output_dict=True)
    classification_report_dfL = pd.DataFrame(classification_report_data).transpose()
    
    # Confusion Matrix
    confusion_matrix_dfL = pd.DataFrame(confusion_matrix(y_test, predL))
    
    
    # ROC
    figL, axL = plt.subplots()
    axL.plot(fprL, tprL, color='darkorange', lw=2, label='LogisticRegression (AUC = %0.2f)' % roc_aucL)
    axL.set_xlim([0.0, 1.0])
    axL.set_ylim([0.0, 1.05])
    axL.set_xlabel('False Positive Rate')
    axL.set_ylabel('True Positive Rate')
    axL.set_title('Receiver operating characteristic')
    axL.legend(loc="lower right")
    
    # Display on Streamlit
    st.write("## Logistic Regression Performance")
    
    col1, col2 = st.columns([1,2])
    with col1:
        st.write("### Confusion matrix")
        st.write(pd.DataFrame(confusion_matrix_dfL))
        st.write("### Evaluate Model")
        st.write(f"Test accuracy = {accL:.4f}")
        st.write(f"Test precision = {precL:.4f}")
        st.write(f"Test recall = {recaL:.4f}")
        st.write("### Classification Report")
        st.dataframe(classification_report_dfL)
    with col2:
        st.write("### ROC curve")
        st.pyplot(figL)
       
with tab3:
    # SVM model
    C = 1 
    kernel = 'linear'
    gamma =  0.1 
    # Train model
    good_modelsvm = svm.SVC(C=C, kernel=kernel, gamma=gamma, probability = True)
    good_modelsvm.fit(X_train,y_train) 
    # Evaluate model
    predsvm = good_modelsvm.predict(X_test) 
    accsvm = accuracy_score(y_test, predsvm) 
    precsvm = precision_score(y_test, predsvm) 
    recasvm = recall_score(y_test, predsvm) 
    
    fpr, tpr, thresholds = roc_curve(y_test, good_modelsvm.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    #Classification Report
    classification_report_data = classification_report(y_test, predsvm, output_dict=True)
    classification_report_dfSVM = pd.DataFrame(classification_report_data).transpose()
    
    # Confusion Matrix
    confusion_matrix_dfSVM = pd.DataFrame(confusion_matrix(y_test, predsvm))
    
    # ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='SVM (AUC = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    # Display on Streamlit
    st.write("## SVM Model Performance")
    col1, col2 = st.columns([1,2])
    with col1:
        st.write("### Confusion Matrix")
        st.dataframe(confusion_matrix_dfSVM)
        st.write("### Classification Report")
        st.write("### Evaluate Model")
        st.write(f"Test accuracy = {accsvm: .4f}")
        st.write(f"Test precision = {precsvm: .4f}")
        st.write(f"Test recall = {recasvm: .4f}")
        st.dataframe(classification_report_dfSVM)
    with col2:
        st.write("### ROC Curve")
        st.pyplot(plt)
        
with tab4:
    # Decision Tree model
    criterion = 'gini'
    max_leaf_nodes = 19
    # Train model
    good_model_D = DecisionTreeClassifier(criterion=criterion,
                                        max_leaf_nodes=max_leaf_nodes) 
    good_model_D.fit(X_train, y_train)
    # Evaluate model
    pred_D = good_model_D.predict(X_test) 
    acc_D = accuracy_score(y_test, pred_D) 
    prec_D = precision_score(y_test, pred_D) 
    reca_D = recall_score(y_test, pred_D)
    
    fpr, tpr, thresholds = roc_curve(y_test, good_model_D.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Classification Report
    classification_report_data = classification_report(y_test, pred_D, output_dict=True)
    classification_report_dfD = pd.DataFrame(classification_report_data).transpose()
    
    # Confusion Matrix
    confusion_matrix_dfD = pd.DataFrame(confusion_matrix(y_test, pred_D))
    
    # ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='DecisionTreeClassifier (AUC = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    
    # Dislay on Streamlit
    st.write("## Decision Tree Model Performance")
    col1, col2 = st.columns([1,2])
    with col1:
        st.write("### Confusion Matrix")
        st.dataframe(confusion_matrix_dfD)
        st.write("### Evaluate Model")
        st.write(f"Test accuracy = {acc_D: .4f}")
        st.write(f"Test precision = {prec_D: .4f}")
        st.write(f"Test recall = {reca_D: .4f}")
        st.write("### Classification Report")
        st.dataframe(classification_report_dfD)
    with col2:
        st.write("### ROC Curve")
        st.pyplot(plt)

with tab5:
    # Random Forest model
    max_features = 15
    max_leaf_nodes = 24
    n_estimators = 50
    # Train model
    good_model = RandomForestClassifier(max_leaf_nodes = max_leaf_nodes,
                                        max_features = max_features, 
                                        n_estimators=n_estimators, ) 
    good_model.fit(X_train, y_train) 
    
    pred = good_model.predict(X_test) # predicted output for test examples
    # Evaluate model
    acc = accuracy_score(y_test, pred) # accuracy on test examples
    prec = precision_score(y_test, pred) # precision on test examples
    reca = recall_score(y_test, pred) 
    
    fpr, tpr, thresholds = roc_curve(y_test, good_model.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # Classification Report
    classification_report_data = classification_report(y_test, pred, output_dict=True)
    classification_report_dfR = pd.DataFrame(classification_report_data).transpose()
    
    # Confusion Matrix
    confusion_matrix_dfR = pd.DataFrame(confusion_matrix(y_test, pred))
    
    # ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='RandomForest (AUC = %0.2f)' % roc_auc)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    
    # Display on Streamlit
    st.write("## Random Forest Model Performance")
    col1, col2 = st.columns([1,2])
    with col1:
        st.write("### Confusion Matrix")
        st.dataframe(confusion_matrix_dfR)
        st.write("### Evaluate Model")
        st.write(f"Test accuracy = {acc: .4f}")
        st.write(f"Test precision = {prec: .4f}")
        st.write(f"Test recall = {reca: .4f}")
        st.write("### Classification Report")
        st.dataframe(classification_report_dfR)
    with col2:
        st.write("### ROC Curve")
        st.pyplot(plt)
