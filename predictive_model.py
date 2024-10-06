#!/usr/bin/env python
# coding: utf-8

# #### Query Page

# In[114]:


import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from datetime import datetime
import random

# Assume this is already trained
def train_behavior_model():
    file_path = r"expanded_dataset.csv"
    df = pd.read_csv(file_path)

    # Define the categories of interest
    categories_of_interest = ['time', 'date', 'weather', 'music']

    # Filter the rows where 'label' is one of the categories in categories_of_interest
    df = df[df['label'].isin(categories_of_interest)]

    # Create 'time_of_day' column by applying the function
    df['time_of_day'] = df['time'].apply(classify_time_of_day)

    # Preprocessing: We are interested in 'time_of_day' and 'command'
    df = df.dropna().reset_index(drop=1)
    X = df[['time_of_day']]  # Features
    y = df['label']          # Labels (actions user took)

    # One-hot encoding the time_of_day
    X = pd.get_dummies(X, drop_first=True)

    # Splitting the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # RandomForest for prediction
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    
    # Evaluate accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # print(f"Model accuracy: {accuracy}")

    # Save the feature names for later use
    return model, X_train.columns

def predict_next_action(model, feature_columns):
    current_time_of_day = categorize_time_of_day()
    # print(f"Current time of day: {current_time_of_day}")

    # Create a DataFrame with one row for the current time of day
    time_of_day_df = pd.DataFrame([[current_time_of_day]], columns=['time_of_day'])
    
    # One-hot encode the time_of_day_df
    time_of_day_df = pd.get_dummies(time_of_day_df, drop_first=True)

    # Ensure all columns are present (create missing columns and set them to 0)
    for col in feature_columns:
        if col not in time_of_day_df.columns:
            time_of_day_df[col] = 0
    
    # Ensure the columns are in the same order as during training
    time_of_day_df = time_of_day_df[feature_columns]

    # print(f"One-hot encoded time_of_day_df:\n{time_of_day_df}")

    # Predict the next action
    predicted_category = model.predict(time_of_day_df)
    return predicted_category[0]

# Function to classify time of day
def classify_time_of_day(time_str):
    time = pd.to_datetime(time_str, format='%H:%M:%S').time()
    hour = time.hour
    if 5 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 17:
        return 'afternoon'
    elif 17 <= hour < 21:
        return 'evening'
    else:
        return 'night'

def categorize_time_of_day():
    current_hour = datetime.now().hour
    if 5 <= current_hour < 12:
        return 'morning'
    elif 12 <= current_hour < 17:
        return 'afternoon'
    elif 17 <= current_hour < 21:
        return 'evening'
    else:
        return 'night'



