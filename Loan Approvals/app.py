
import pickle
import streamlit as st
import math
import pandas as pd
import numpy as np

pickle_in = open('loanApproval', 'rb')
knn, lr_model, features_means, features_std, features = pickle.load(pickle_in)

@st.cache()

# Define the function which will make the prediction using data
# inputs from users

def prediction(classifier, debt, marrital_status, bank_customer, employment_type, prior_default, employment, drivers_license,
               citizenship, years_employed, credit_score, income):
    
    #Populating the booleans
    marrital_status = 1 if marrital_status == 'Yes' else 0
    bank_customer = 1 if bank_customer == 'Yes' else 0
    prior_default = 0 if prior_default == 'Yes' else 1    #prior_defaulters are identified as 0 in the dataset
    employment = 1 if employment == 'Yes' else 0
    drivers_license = 1 if drivers_license == 'Yes' else 0
    
    #taking log transform
    years_employed = np.log(years_employed)
    credit_score = np.log(credit_score)
    income = np.log(income)
    
    #initialization
    employment_types = {'Industrial':0,
                'Materials':0,
                'Consumer Services':0,
                'Healthcare':0,
                'Financials':0,
                'Education':0,
                'Utilities':0}
    
    citizenships = {'Citizen By Birth':0,
                    'Temporary Citizenship':0,
                    'Other':0}
    
    #value population
    for key in employment_types:
        if key == employment_type:
            employment_types[key] = 1
            
    for key in citizenships:
        if key == citizenship:
            citizenships[key] = 1
            
    X_test_df = pd.DataFrame([[debt, marrital_status, bank_customer, 
                               employment_types.get('Industrial'),employment_types.get('Materials'),employment_types.get('Consumer Services'),
                               employment_types.get('Healthcare'),employment_types.get('Financials'),employment_types.get('Education'),
                               employment_types.get('Utilities'),
                               prior_default, employment, drivers_license,
                               citizenships.get('Citizen By Birth'), citizenships.get('Temporary Citizenship'),citizenships.get('Other'),
                               years_employed, credit_score, income]], 
        columns=features)
    
    #normalise the test data
    X_test_df = (X_test_df - features_means)/features_std
    
    # Make predictions
    message = ''
    if classifier == 'Logistic Regression':
        prediction = lr_model.predict_proba(X_test_df)
        THRESHOLD = 0.8
    else: 
        prediction = knn.predict_proba(X_test_df)
        THRESHOLD = 0.6
        message = 'AUC for Log Reg is more than AUC for kNN. Accuracy is also better for log reg thus prefer Log Reg'

    if prediction[0][1] > THRESHOLD:
        return ['Your loan is approved :)',message]
    else: 
        return ['Sorry your loan is rejected :(',message]

# This is the main function in which we define our webpage
def main():
    
    st.title('Jarvis Loan Approval System')
    # Create input fields
    classifier = st.radio("Which model you want to use for prediction?",
                            ('kNN','Logistic Regression'))
    
    marrital_status = st.radio('Is the customer married?',
                             ('Yes','No'))
    
    bank_customer = st.radio('Is the borrower already bank customer',
                             ('Yes','No'))
    
    prior_default = st.radio('Is the borrower prior defaulter?',
                             ('Yes','No'))
    
    employment = st.radio('Is the borrower employed?',
                             ('Yes','No'))
    
    drivers_license = st.radio('Does the borrower have drivers license?',
                             ('Yes','No'))
    
    debt = st.number_input("How much debt does borrower already have in $1000s?",
                                  min_value=0.000,
                                  max_value=30.000,
                                  value=10.000,
                                  step=2.500,
                                 )
    
    employment_type = st.selectbox('Select the employment type',
                             ('Industrial','Materials','Consumer Services','Healthcare','Financials','Utilities','Education'))
    
    citizenship = st.selectbox('Does borrower have US citizenship?',
                             ('Citizen By Birth','Temporary Citizenship','Other'))
               
    years_employed = st.number_input("For how many years the borrower has been employed?",
                                  min_value=1.0,
                                  max_value=30.0,
                                  value=10.0,
                                  step=1.0,
                                 )
    
    credit_score = st.number_input("What is the credit score of borrower?",
                                  min_value=1.0,
                                  max_value=70.0,
                                  value=10.0,
                                  step=5.0,
                                 )
    
    income = st.number_input("What is the income of borrower?",
                                  min_value=1,
                                  max_value=50000,
                                  value=30000,
                                  step=5000,
                                 )
               
    result = ""
    
    # When 'Predict' is clicked, make the prediction and store it
    if st.button("Should be approve the loan?"):
        result = prediction(classifier, debt, marrital_status, bank_customer, employment_type, prior_default, employment, drivers_license,
               citizenship, years_employed, credit_score, income)
        st.success(result[0])
        if classifier == 'kNN':
            st.success(result[1])
        
if __name__=='__main__':
    main()
