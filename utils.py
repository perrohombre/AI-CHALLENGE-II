import streamlit as st
import pandas as pd
import requests
import ast
from dotenv import load_dotenv
import os
from playsound import playsound
import random
import time

load_dotenv()

# Replace these with your actual API key and endpoint
AZURE_API_KEY = os.getenv('AZURE_COMPLETIONS_API_KEY')
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_COMPLETIONS_OPENAI_ENDPOINT')


def analyze_sentiment_customer(response_text):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a sentiment analysis expert. Please respond with only a number."},
            {"role": "user", "content": f"Rate the sentiment of this response on a scale from 1 (negative) to 100 (positive). Respond only with a number: {response_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 5,
        "top_p": 0.95
    }

    try:
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        sentiment_score = response.json()["choices"][0]["message"]["content"].strip()
        sentiment_score = int(sentiment_score)
    except (requests.RequestException, ValueError) as e:
        print(f"Error during sentiment analysis: {e}")
        if "429" in str(e):
            time.sleep(10)
            sentiment_score = analyze_sentiment_customer(response_text)
        else:
            sentiment_score = random.randint(1, 100)

    return sentiment_score

def analyze_sentiment_salesman(response_text):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }

    payload = {
        "messages": [
            {"role": "system", "content": "You are a sentiment analysis expert. Please respond with only a number."},
            {"role": "user", "content": f"Rate the sentiment of this response on a scale from 1 (negative) to 100 (positive). Keep in mind that this is a salesman response, so be very strict; don't hesitate to give a negative mark. Respond only with a number: {response_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 5,
        "top_p": 0.95
    }

    try:
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        sentiment_score = response.json()["choices"][0]["message"]["content"].strip()
        sentiment_score = int(sentiment_score)
    except (requests.RequestException, ValueError) as e:
        print(f"Error during sentiment analysis: {e}")
        if "429" in str(e):
            time.sleep(10)
            sentiment_score = analyze_sentiment_salesman(response_text)
        else:
            sentiment_score = random.randint(1, 100)
    
    return sentiment_score

def extract_customer_responses(conversation):
    customer_responses = []
    for message in conversation:
        if message['role'] == 'customer':
            customer_responses.append(message['text'])
    return customer_responses

def extract_salesman_responses(conversation):
    salesman_responses = []
    for message in conversation:
        if message['role'] == 'salesman':
            salesman_responses.append(message['text'])
    return salesman_responses

@st.cache_data
def load_data():
    df = pd.read_csv("assets/telecom_1000.csv")  # Ensure the CSV file is in the correct path
    df['conversation'] = df['conversation'].apply(ast.literal_eval)  # Convert string to list
    return df

def get_feedback_for_salesman(salesman_name, responses_with_scores):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_API_KEY
    }

    responses_text = '\n'.join([f"Response: {resp}\nSentiment Score: {score}" for resp, score in responses_with_scores])
    prompt = f"As a sales coach, provide feedback for salesman {salesman_name} based on the following responses and their sentiment scores:\n{responses_text}\n\nJudge the responses strictly and provide constructive feedback. Provide only one overall feedback for all responses. If sentiments are high, don't hesitate to praise. Feedback should be maximum 4 sentences long."

    payload = {
        "messages": [
            {"role": "system", "content": "You are an expert sales coach."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 150,
        "top_p": 0.95
    }

    try:
        response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()

        feedback = response.json()["choices"][0]["message"]["content"].strip()
    except (requests.RequestException, ValueError) as e:
        feedback = "Error generating feedback."
        st.write(f"Error during feedback generation: {e}")

    return feedback

def read_answer(text, voice_name="alloy"):
    print("Reading")
    # Azure OpenAI TTS Endpoint and API Key
    tts_endpoint = "https://aict-m2n5g2gx-northcentralus.openai.azure.com/openai/deployments/tts-hd/audio/speech?api-version=2024-05-01-preview"
    api_key = "8c5c55e2fa3846c3a153dafe8ca364f6"

    #---tts1

    tts_endpoint = "https://aict-m2n5g2gx-northcentralus.openai.azure.com/openai/deployments/tts/audio/speech?api-version=2024-05-01-preview"
    api_key = "8c5c55e2fa3846c3a153dafe8ca364f6"

    # Headers for TTS API
    headers = {
        "api-key": api_key,
        "Content-Type": "application/json",
    }

    # Payload including required fields
    payload = {
        "model": "tts",  # Specify the model name
        "input": text,  # Input text as a string
        "voice": voice_name  # Specify the voice
    }

    # Send POST request to TTS API
    response = requests.post(tts_endpoint, headers=headers, json=payload, timeout=10)

    # Check if synthesis was successful
    if response.status_code == 200:
        with open("output.mp3", "wb") as audio_file:
            audio_file.write(response.content)
            playsound("output.mp3")
        print("Speech synthesis completed successfully.")
    else:
        print(f"Error {response.status_code}: {response.text}")