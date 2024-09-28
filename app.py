#!/usr/bin/env python
# coding: utf-8

# ### Import all the required libraries
import streamlit as st
from streamlit_navigation_bar import st_navbar
import plotly.express as px
from datetime import datetime
import time
import plotly.graph_objects as go


import speech_recognition as sr
import pyttsx3

import os
import webbrowser
import requests
# import datetime

import playsound
import pygame
from gtts import gTTS

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from nltk.sentiment import SentimentIntensityAnalyzer

import random
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


from suggestive_model import train_behavior_model,predict_next_action


# ### Creating all the required functions



# Initialize the recognizer and text-to-speech engine
engine = pyttsx3.init()




def speak(text):
    tts = gTTS(text=text, lang='en')
    filename = "temp_audio.mp3"
    tts.save(filename)

    # Initialize pygame mixer
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(text))
    pygame.mixer.music.play()

    # Wait until the sound finishes playing
    while pygame.mixer.music.get_busy():
        continue

    pygame.mixer.quit()
    os.remove(filename)  



def listen():
    recognizer = sr.Recognizer()
    
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source)  # Reduce background noise
        try:
            # Increase the timeout and phrase time limit to give more time to speak
            audio = recognizer.listen(source)
            command = recognizer.recognize_google(audio)  # Use Google's speech recognition
            with st.chat_message("user"):
              st.markdown(command)
            print(f"User said: {command}")
            return command.lower()
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that. Could you please repeat?")
            return "None"
        except sr.RequestError:
            print("Sorry, there was an issue with the speech recognition service.")
            return "None"



def get_time():
    now = datetime.now()
    time_str = now.strftime("%H:%M:%S")
    return time_str

def get_date():
    today = datetime.today()
    date_str = today.strftime("%B %d, %Y")
    return date_str





def save_command_to_csv(command, category):
    file_path = r"expanded_dataset.csv"
    date_str = get_date()
    date_obj = datetime.strptime(date_str, '%B %d, %Y')
    formatted_date = date_obj.strftime('%d-%m-%Y')
    # Create a new dataframe for the current command and category
    new_data = pd.DataFrame([{"command": command, "label": category,"date":formatted_date,"time":get_time()}])
    
    # Check if the CSV file already exists
    if os.path.exists(file_path):
        # If it exists, append the new data to it
        try:
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        except pd.errors.EmptyDataError:
            updated_data = new_data  # If the file is empty, initialize with new data
    else:
        updated_data = new_data  # If file doesn't exist, initialize with new data
    
    # Save the updated data back to the CSV
    updated_data.to_csv(file_path, index=False)




def open_website(command):
    websites = {
        "google": "https://www.google.com",
        "youtube": "https://www.youtube.com",
        "flipkart": "https://www.flipkart.com",
        "amazon": "https://www.amazon.in",
        "gmail": "https://mail.google.com",
    }
    for site_name, url in websites.items():
        if site_name in command:
            webbrowser.open(url)
            return f"Opening {site_name}"
    query = command.replace("search", "").strip()
    if query:
            speak(f"Searching for {query} on the web.")
            webbrowser.open(f"https://www.google.com/search?q={query}")
    return "Please wait, I am searching!"



def get_weather(city_name):
    api_key = "dd595ee9637f42dfafd170114242309"  
    base_url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city_name}"

    response = requests.get(base_url)
    weather_data = response.json()
    
    if "error" not in weather_data:
        current_weather = weather_data["current"]
        temperature = current_weather["temp_c"]
        weather_desc = current_weather["condition"]["text"]
        wind = current_weather["wind_kph"]
        cloud = current_weather["cloud"]
        feels_like = current_weather["feelslike_c"]
        is_day = current_weather["is_day"]
        humidity = current_weather["humidity"]
        
        time_of_day = "daytime" if is_day else "night"
        
        weather_info=( f"The current temperature in {city_name} is {temperature}°C during {time_of_day} with {weather_desc}. "
        f"It feels like {feels_like}°C, with wind speeds of {wind} kilometers per hour. "
        f"Cloud cover is {cloud}%, and the humidity is {humidity}%.")    

        print(weather_info)
        return weather_info
    else:
        return "City not found. Please try again."


# ### Creating Machine Learning model
# ##### This will predict which command belongs to which categories.


def train_and_predict(command):
    # Load the dataset
    file_path = r"expanded_dataset.csv"
    df = pd.read_csv(file_path)
    df = df.dropna()
    # Separate features and labels
    X = df['command']
    y = df['label']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Vectorize the text data using CountVectorizer
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Initialize the classifier
    classifier = MultinomialNB()

    # Train the classifier
    classifier.fit(X_train_vec, y_train)

    # Make predictions on the test set and evaluate the model
    y_pred = classifier.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    # report = classification_report(y_test, y_pred)

    # print(f"Accuracy: {accuracy:.2f}")
    # print("Classification Report:\n", report)

    # Now predict the category for the given command
    # Vectorize the new sentence
    new_command_vec = vectorizer.transform([command])

    # Predict the category for the new command
    predicted_category = classifier.predict(new_command_vec)

    # Return the predicted category
    # print(predicted_category[0])
    return predicted_category[0]

# Function to load cities from a file
def load_cities_from_file(filename='cities.txt'):
    with open(filename, 'r') as file:
        cities = file.read().splitlines()  # Reads each line as a city name
    return cities

# Load the cities list once at the start
cities_list = load_cities_from_file()
cities_list = [city.strip().lower() for city in cities_list]


def process_command(command):
    command_category=train_and_predict(command)
    save_command_to_csv(command, command_category)
    if 'date' in command and 'time' in command:
        current_time=get_time()
        current_date=get_date()
        speak(f"Today is {current_date} and the time now is {current_time}")
    elif command_category=='time':
        current_time = get_time()
        speak(f"The time is {current_time}")
    elif command_category=='date':
        current_date = get_date()
        speak(f"Today's date is {current_date}")
    elif command_category=='weather':
        city_in_string = command.lower().split()
        matched_city = next((city for city in city_in_string if city in cities_list), None)
        if matched_city:
            weather_info = get_weather(matched_city)
            speak(weather_info)
        else:
            speak("Which city?")
            city = listen()
            if city != "None":
                weather_info = get_weather(city)
                speak(weather_info)
    elif command_category == 'search':
        query = command.replace("search", "").strip()
        if query:
            speak(f"Searching for {query} on the web.")
            webbrowser.open(f"https://www.google.com/search?q={query}")
        else:
            speak("What would you like to search for?")
    elif command_category == 'music':
        song_query = command.replace("play", "").strip()  
        if song_query:
            speak(f"Playing {song_query} on YouTube.")
            webbrowser.open(f"https://www.youtube.com/results?search_query={song_query}")
        else:
            speak("What song would you like to play?")     
    elif command_category=='open':
        if 'notepad' in command:
            speak("Opening Notepad")
            print("Opening Notepad")
            os.system('notepad')
        elif 'calculator' in command:
            speak("Opening Calculator")
            print("Opening Calculator")
            os.system('calc')
        response = open_website(command)
        speak(response)
    else:
        speak("Sorry, I don't know how to respond to that.")



def analyze_sentiment(text):
    """Analyze sentiment using SentimentIntensityAnalyzer and return the compound score."""
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment['compound']  # Returns a value between -1 and 1





def ask_how_are_you():
    speak("How are you feeling today?")
    print("How are you feeling today?")
    user_feeling = listen()
    
    # Analyze sentiment
    sentiment_score = analyze_sentiment(user_feeling)
    
    if sentiment_score < 0:  # Very negative
        speak("I hope things get better for you. Do you want to relax with some music?")
        play_song_recommendation()
        # print("I'm sorry to hear that. Would you like to talk about it?")
    # elif -0.5 <= sentiment_score < 0:  # Negative
    #     speak("I hope things get better for you. Do you want to relax with some music?")
    #     # print("I hope things get better for you. Do you want to relax with some music?")
    #     play_song_recommendation()
    elif 0 <= sentiment_score < 0.5:  # Neutral
        speak("Thanks for sharing! Is there anything specific I can help you with?")
        # print("Thanks for sharing! Is there anything specific I can help you with?")
        
    else:  # Very positive
        speak("I'm glad to hear you're feeling great! Anything exciting planned today?")
        print("I'm glad to hear you're feeling great! Anything exciting planned today?")



def play_song_recommendation():
    recommendations = [
        "Can I play your favorite song to cheer you up?",
        "Would you like to listen to some music?",
        "How about some calming tunes?",
        "I can play a cheerful song for you!"
    ]
    faviroute_songs = [
        "https://www.youtube.com/watch?v=0Pu8KCya9YY&list=RDMM&start_radio=1&rv=cl0a3i2wFcc",
        "https://www.youtube.com/watch?v=OY5vL4aXMAo&list=RDMM&index=3",
        "https://youtu.be/lbCRtrrMvSw?si=hzw8cjZF_nTljkmr",
        "https://www.youtube.com/watch?v=HHgVlMrkloQ&list=PL0bucKqfv5E2-Dn83UgFkKkUl2QtpupqQ&index=27",
        "https://www.youtube.com/watch?v=we_U_WNM4Zk&list=PL0bucKqfv5E2-Dn83UgFkKkUl2QtpupqQ&index=28",
        "https://www.youtube.com/watch?v=bESWkKFsKZE",
        "https://www.youtube.com/watch?v=QYO6AlxiRE4&list=RDMM&index=5"
    ]
    speak(random.choice(recommendations))
    response = listen()
    if 'yes' in response or 'sure' in response or 'ok' in response:
        speak("Great! Let me play something for you.")
        print("Great! Let me play something for you.")
        webbrowser.open(random.choice(faviroute_songs))
    else:
        speak("Alright, let me know if you need anything.")
        print("Alright, let me know if you need anything.")

### Running the main function

def main():
    # speak("Hello Sir, I am your assistant. How can I help you today?")
    ask_how_are_you()  # Ask how the user is feeling
    while True:
        command = listen()
        if command == "None":
            continue
        if 'exit' in command or 'stop' in command:
            speak("Goodbye Sir, Hope you will need me again!")
            print("Goodbye Sir, Hope you will need me again!")
            break
        else:
            process_command(command)





























#### Creating Streamlit Interface

def response_generator(response:str):
    for word in response.split():
        yield word + " "
        time.sleep(0.05)

def date_generator():
    yield datetime.now().strftime('%H:%M %p')
    time.sleep(0.05)


st.set_page_config(initial_sidebar_state="collapsed")

styles = {
    "nav": {
        "background-color": "#0e1117",
    },
    "div": {
        "max-width": "32rem",
    },
    "span": {
        "border-radius": "0.5rem",
        "color": "#EEEEEE",
        "margin": "0 0.125rem",
        "padding": "0.4375rem 0.625rem",
    },
    "active": {
        "background-color": "rgba(255, 255, 255, 0.25)",
    },
    "hover": {
        "background-color": "rgba(255, 255, 255, 0.35)",
    },
}

# Page navigation
pages = ["Home", "Voice Assistant", "User Activity Logs","Predictive Behaviour"]

page = st_navbar(pages, styles=styles)


# Placeholder to store user activities
if "activity_log" not in st.session_state:
    st.session_state["activity_log"] = []


# Page 1: Home Page
if page == "Home":
    st.title("Welcome to Your Voice Assistant")

    st.markdown("""
    This is your personal AI-powered voice assistant capable of performing a variety of tasks like:
    - **Playing music** or **searching for content** online.
    - **Opening applications** like Notepad or Calculator.
    - **Telling the date and time**.
    - **Providing weather updates** for any city.
    - And much more!
    
    Feel free to navigate to the **Voice Assistant** page to start giving commands or check your interaction history on the **User Activity Logs** page.
    """)

    # Display a live clock with current date and time
    current_time = st.empty()
    current_date = st.empty()

    while True:
        with current_time:
            st.write(f"**Current Time:** {datetime.now().strftime('%H:%M:%S %p')}")
        with current_date:
            st.write(f"**Current Date:** {datetime.now().strftime('%A, %B %d, %Y')}")
        time.sleep(1)  


# Page 2: Voice Assistant Interface
if page == "Voice Assistant":
    st.title("Voice Assistant")

    # Create a button to start the voice assistant
    start_assistant = st.button("Start Voice Assistant 🎙️")

    if start_assistant:
        st.text("Listening... 🎙️")

        # Run the voice assistant (main function) when button is clicked
        while True:
            with st.spinner('Listening...'):
                user_command = main()

            if user_command is None:
                user_command = "None"

            else:
                with st.chat_message("user"):
                    st.markdown(user_command)         

            if user_command == "None":
                response = "Thank You for your time"
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(response))
                speak(response) 
                break  
            else:
                response = process_command(user_command)
                with st.chat_message("assistant"):
                    response = st.write_stream(response_generator(response))
                speak(response)


# Page 3: User Activity Logs
if page == "User Activity Logs":
    st.title("User Activity Logs")

    file_path = r"expanded_dataset.csv"

    df = pd.read_csv(file_path)

    # # Load the JSON data (mock data or load from file)
    # with open('data.json', 'r') as f:
    #     data = json.load(f)

    df = df[::-1]

    # logs_table = st.table(df)
    st.dataframe(df, use_container_width=True)



    labels = df['label'].unique().tolist()
    values = df['label'].value_counts().reindex(labels).tolist()

    plot_df = pd.DataFrame({
        'label': labels,
        'count': values,
        'percentage': [round((x / sum(values)) * 100, 2) for x in values]
    })

    # # fig = go.Figure(data=[go.Pie(labels=labels, values=values)])
    fig = px.pie(plot_df, values='count', names='label',
                title='Distribution of User Commands by Category',
                hover_data=['count'], labels={'count':'No. of commands'})
    fig.update_traces(textposition='inside', textinfo='percent+label')

    st.plotly_chart(fig)

 
# Page 4: Predictive Behaviour
if page == "Predictive Behaviour":
    st.title("Predictive Behaviour")

    # Inform the user what is happening
    st.write("The system is analyzing your previous interactions...")
    speak("The system is analyzing your previous interactions...")

    # Train the model
    model, feature_columns = train_behavior_model()

    # Predict the next action
    predicted_action = predict_next_action(model, feature_columns)

    # Display the predicted action immediately
    # st.write(f"**Predicted next action is:** {predicted_action}")
    speak(f"Predicted next action is: {predicted_action}")

    # Optionally, perform the predicted action automatically if needed
    if predicted_action == "music":
        speak("Playing your favorite music!")
        webbrowser.open("https://www.youtube.com/watch?v=lbCRtrrMvSw")
    elif predicted_action == "weather":
        speak("Which city?")
        city = listen()
        if city != "None":
            weather_info = get_weather(city)
            speak(weather_info)
    elif predicted_action == "time":
        current_time = get_time()
        speak(f"The current time is {current_time}")
    # Add more predicted actions based on your dataset and commands
    elif predict_next_action == "date":
        current_date = get_date()
        speak(f"Today's date is {current_date}")
    else:    
        st.write("No specific action predicted.")
        speak("No specific action predicted.")
