import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score

# Title for the application
st.title('INX Employees Performance Rating and Factors Influencing Growth Rate')

# Brief introduction
st.write('''In recent years, we have witnessed a decline in employee performance at INX Future, a center of data analytics and automation solutions. 
This decline has led to increased service delivery escalations and a noticeable drop in client satisfaction. 
The company initiated this project to identify key drivers of performance and address the challenges it is facing. 
This analysis will help the company make informed decisions in the future.''')

# Importing the dataset
df = pd.read_csv("INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8 (1).csv")

# Checking the shape of the dataset
st.write('Viewing the number of rows and columns:', df.shape)

# Displaying the first 5 rows
st.write(df.head(5))

# Columns to encode
columns_encoded = ['EmpNumber', 'Gender', 'EducationBackground', 'MaritalStatus', 'EmpDepartment', 'EmpJobRole', 'BusinessTravelFrequency', 'OverTime', 'Attrition']

# Encoding categorical variables
le_dict = {col: LabelEncoder() for col in columns_encoded}
for col in columns_encoded:
    le_dict[col].fit(df[col])
    df[col] = le_dict[col].transform(df[col])

# Encode target variable
le_target = LabelEncoder()
df['PerformanceRating'] = le_target.fit_transform(df['PerformanceRating'])

# Features and target variable
# Defining the features (X) and the target variable (y)
X = df[['EmpDepartment', 'EmpJobRole', 'EmpEnvironmentSatisfaction', 'EmpLastSalaryHikePercent', 'TotalWorkExperienceInYears', 'EmpWorkLifeBalance', 'ExperienceYearsAtThisCompany', 'ExperienceYearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']]
y = df['PerformanceRating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# View the shape of train and test sets
print('y_train.shape:', y_train.shape)
print('y_test.shape:', y_test.shape)
print('X_train.shape:', X_train.shape)
print('X_test.shape:', X_test.shape)

# Initialize the Decision Tree Classifier
clf = DecisionTreeClassifier()

# Train the classifier
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')

# Print the evaluation metrics
print("Accuracy:", accuracy)
print("F1 Score:", f1)
print("Precision:", precision)


# Sidebar for user input
st.sidebar.write("Enter New Data For Prediction")

Employee_Department = st.sidebar.selectbox('EmpDepartment', le_dict['EmpDepartment'].classes_)
Education_Background = st.sidebar.selectbox('EducationBackground', le_dict['EducationBackground'].classes_)
Emp_Job_Role = st.sidebar.selectbox('EmpJobRole', le_dict['EmpJobRole'].classes_)
Business_Travel_Frequency = st.sidebar.selectbox('BusinessTravelFrequency', le_dict['BusinessTravelFrequency'].classes_)
OverTime = st.sidebar.selectbox('OverTime', le_dict['OverTime'].classes_)
MaritalStatus = st.sidebar.selectbox('MaritalStatus', le_dict['MaritalStatus'].classes_)
EmpLastSalaryHikePercent = st.sidebar.number_input('EmpLastSalaryHikePercent')
ExperienceYearsAtThisCompany = st.sidebar.number_input('ExperienceYearsAtThisCompany')
YearsSinceLastPromotion = st.sidebar.number_input('YearsSinceLastPromotion')
ExperienceYearsInCurrentRole = st.sidebar.number_input('ExperienceYearsInCurrentRole')

# Encoding for User Input
encoded_input = [
    le_dict['EmpDepartment'].transform([Employee_Department])[0],
    le_dict['EducationBackground'].transform([Education_Background])[0],
    le_dict['EmpJobRole'].transform([Emp_Job_Role])[0],
    le_dict['BusinessTravelFrequency'].transform([Business_Travel_Frequency])[0],
    le_dict['OverTime'].transform([OverTime])[0],
    le_dict['MaritalStatus'].transform([MaritalStatus])[0],
    EmpLastSalaryHikePercent,
    ExperienceYearsAtThisCompany,
    YearsSinceLastPromotion,
    ExperienceYearsInCurrentRole
]

# Convert to DataFrame
encoded_input_df = pd.DataFrame([encoded_input])

# Predicting the performance rating
if st.sidebar.button('Predict Performance Rating'):
    prediction = clf.predict(encoded_input_df)[0]
    st.sidebar.write('Predicted Performance Rating:', le_target.inverse_transform([prediction])[0])


