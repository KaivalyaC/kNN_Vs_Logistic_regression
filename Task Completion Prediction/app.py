
import pickle
import streamlit as st
import math
import pandas as pd
import numpy as np

pickle_in = open('taskCompletionPrediction', 'rb')
knn,lr_model,features_means,features_std,features = pickle.load(pickle_in)

@st.cache()

def prediction(classifier, employee_experience, training_level):
    
    #initialization
    training_levels = {'Traing Level 4':0,
                'Traing Level 6':0,
                'Traing Level 8':0}
    
    #value population
    for level in training_levels:
        if level == training_level:
            training_levels[level] = 1
            
    X_test_df = pd.DataFrame([[employee_experience,
            training_levels.get('Traing Level 4'),
            training_levels.get('Traing Level 6'),
            training_levels.get('Traing Level 8')]], 
        columns=features)
    
    #normalise the test data
#     features_means = getFeatureMeans()
#     features_std = getFeatureStd()
    X_test_df = (X_test_df - features_means)/features_std
    
    # Make predictions
    message = ''
    if classifier == 'Logistic Regression':
        prediction = lr_model.predict_proba(X_test_df)
        THRESHOLD = 0.5
        message = 'For logistic Reg, train accuracy is 0.92 where as test accuracy is 0.80. It shows signs of overfitting so use kNN instead'
    else: 
        prediction = knn.predict_proba(X_test_df)
        THRESHOLD = 0.5
    
    if prediction[0][1] > THRESHOLD:
        return ['Task will be completed', message]
    else: 
        return ['Task will not becompleted', message]

# This is the main function in which we define our webpage
def main():
    
    st.title('Jarvis Task Completion Predictor')
    # Create input fields
    classifier = st.radio("Which model you want to use for prediction?",
                            ('kNN','Logistic Regression'))
    
    employee_experience = st.number_input("How experienced is the Employee?",
                                  min_value=2.0,
                                  max_value=14.0,
                                  value=7.0,
                                  step=0.5,
                                 )
               
    training_level = st.selectbox('Select the training level of employee',
                             ('Traing Level 4','Traing Level 6','Traing Level 8'))
    
    result = ""
    
    # When 'Predict' is clicked, make the prediction and store it
    if st.button("Will the task be completed?"):
        result = prediction(classifier, employee_experience, training_level)
        st.success(result[0])
        if classifier == 'Logistic Regression':
            st.success(result[1])
        
if __name__=='__main__':
    main()
