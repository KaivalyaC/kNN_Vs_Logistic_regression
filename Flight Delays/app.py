
import pickle
import streamlit as st
import math
import pandas as pd
import numpy as np

pickle_in = open('flightDealy', 'rb')
knn, lr_model, features_means, features_std, features = pickle.load(pickle_in)

@st.cache()

# Define the function which will make the prediction using data
# inputs from users

def prediction(classifier, scheduled_departure_time, carrier, destination, distance,
               origin, weather, day):
    
    #initialization
    carriers = {'Delta':0,
                'US':0,
                'Envoy':0,
                'Continental':0,
                'Discovery':0,
                'Others':0}
    
    destinations = {'John F. Kennedy International Airport':0,
                    'Newark Liberty International Airport':0,
                    'LaGuardia Airport':0}
    
    origins = {'Ronald Reagan Washington National Airport':0,
                'Washington Dulles International Airport':0,
                'Baltimore/Washington International Thurgood Marshall Airport':0}
    
    days = {'Monday':0,
            'Tuesday':0,
            'Wednesday':0,
            'Thursday':0,
            'Friday':0,
            'Saturday':0,
            'Sunday':0}
               
    weather_list = {
        'Good':0,
        'Bad':1
    }    
    
    #value population
    for key in carriers:
        if key == carrier:
            carriers[key] = 1
            
    for key in destinations:
        if key == destination:
            destinations[key] = 1
            
    for key in origins:
        if key == origin:
            origins[key] = 1
               
    for key in days:
        if key == day:
            days[key] = 1
    
    X_test_df = pd.DataFrame([[scheduled_departure_time,
          carriers.get('Delta'),carriers.get('US'),carriers.get('Envoy'),carriers.get('Continental'),carriers.get('Discovery'),carriers.get('Others'),
          destinations.get('John F. Kennedy International Airport'),
          destinations.get('Newark Liberty International Airport'),
          destinations.get('LaGuardia Airport'),
          distance,
          origins.get('Ronald Reagan Washington National Airport'),
          origins.get('Washington Dulles International Airport'),
          origins.get('Baltimore/Washington International Thurgood Marshall Airport'),
          weather_list.get(weather),
          days.get('Monday'),days.get('Tuesday'),days.get('Wednesday'),days.get('Thursday'),days.get('Friday'),days.get('Saturday'),days.get('Sunday')
         ]], 
        columns=features)
    
    #normalise the test data
    X_test_df = (X_test_df - features_means)/features_std
    
    # Make predictions
    if classifier == 'Logistic Regression':
        prediction = lr_model.predict_proba(X_test_df)
        THRESHOLD = 0.22
    else: 
        prediction = knn.predict_proba(X_test_df)
        THRESHOLD = 0.2
    
    if prediction[0][1] > THRESHOLD:
        return 'Sorry the flight is delayed'
    else: 
        return 'Your flight is on time'

# This is the main function in which we define our webpage
def main():
    
    st.title('Jarvis Flight Delay Predictor')
    # Create input fields
    classifier = st.radio("Which model you want to use for prediction?",
                            ('kNN','Logistic Regression'))
    
    scheduled_departure_time = st.number_input("What is the scheduled time of departure?",
                                  min_value=0.00,
                                  max_value=24.00,
                                  value=12.50,
                                  step=0.50,
                                 )
               
    carrier = st.selectbox('Select the flight carrier',
                             ('Delta','US','Envoy','Continental','Discovery','Others'))
    
    destination = st.selectbox('Select the destination',
                             ('John F. Kennedy International Airport', 'Newark Liberty International Airport', 'LaGuardia Airport'))
    
    distance = st.number_input("What is the distance traveled",
                                  min_value=150,
                                  max_value=240,
                                  value=200,
                                  step=10,
                                 )
    
    origin = st.selectbox('Select the origin',
                             ('Ronald Reagan Washington National Airport','Washington Dulles International Airport','Baltimore/Washington International Thurgood Marshall Airport'))
    
    weather = st.selectbox('How is the weather',
                             ('Good','Bad'))
               
    day = st.selectbox('Select the day of flight',
                             ('Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'))
               
    result = ""
    
    # When 'Predict' is clicked, make the prediction and store it
    if st.button("Is the flight delayed?"):
        result = prediction(classifier, scheduled_departure_time, carrier, destination, distance,
               origin, weather, day)
        st.success(result)
        
if __name__=='__main__':
    main()
