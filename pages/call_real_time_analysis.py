import streamlit as st

import pyaudio
import wave
import numpy as np
from openai import AzureOpenAI
import random
import requests
import os
from playsound import playsound
import pandas as pd
import matplotlib.pyplot as plt
import threading
from utils import read_answer



from utils import (
    analyze_sentiment_customer,
    analyze_sentiment_salesman,
    extract_customer_responses,
    extract_salesman_responses,
    load_data,
    get_feedback_for_salesman
)

# Salesman data
salesmen = [
    # {"id": 1, "name": "John", "character": "patient and understanding"},
    {"id": 2, "name": "Alice", "character": "quick and efficient"},
    # {"id": 3, "name": "Bob", "character": "formal and professional"},
    # {"id": 4, "name": "Carol", "character": "casual and friendly"},
    # {"id": 5, "name": "David", "character": "impatient and rude"},
    # {"id": 6, "name": "Eve", "character": "unhelpful and dismissive"},
    # {"id": 7, "name": "Frank", "character": "straightforward and direct"},
    # {"id": 8, "name": "Grace", "character": "soft-spoken and polite"},
    # {"id": 9, "name": "Hank", "character": "distracted and uninterested"},
    # {"id": 10, "name": "Ivy", "character": "energetic and enthusiastic"},
    # {"id": 11, "name": "Jack", "character": "confused and uncertain"},
    # {"id": 12, "name": "Kelly", "character": "pushy and aggressive"},
    # {"id": 13, "name": "Leo", "character": "helpful and supportive"},
    # {"id": 14, "name": "Mia", "character": "sarcastic and witty"},
    # {"id": 15, "name": "Nick", "character": "perfect salesperson"},
    # {"id": 16, "name": "Olivia", "character": "charming and persuasive"},
    # {"id": 17, "name": "Peter", "character": "knowledgeable and informative"},
    # {"id": 18, "name": "Quinn", "character": "extremly rude, hates customers"},
    # {"id": 19, "name": "Rachel", "character": "confident and assertive"},
    # {"id": 20, "name": "Sam", "character": "funny and entertaining"}
]

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
SILENCE_THRESHOLD = 500
SILENCE_DURATION = 4

random_index = random.randint(0, len(salesmen) - 1)
NAME = salesmen[random_index]['name']
CHARACTER = salesmen[random_index]['character']


# Initialize application
sentiment_results = []
all_customer_sentiments = []


def analyze_sentiment_real_time(response):

    print("Analizuje sentyment...")

    if 'sentiment_results' not in st.session_state:
        st.session_state['sentiment_results'] = []
        st.session_state['all_customer_sentiments'] = []
        st.session_state['fig2'], st.session_state['ax2'] = plt.subplots()

    # Analiza odpowiedzi klienta
    sentiment_score = analyze_sentiment_customer(response)  # Funkcja, która analizuje sentyment
    st.session_state['sentiment_results'].append({
        'question_id': len(st.session_state['sentiment_results']) + 1,
        'customer_response': response,
        'sentiment_score': sentiment_score
    })
    st.session_state['all_customer_sentiments'].append(sentiment_score)

    sentiment_df = pd.DataFrame(st.session_state['sentiment_results'])
    st.session_state['rt_sentiment_df'] = sentiment_df

    overall_avg_customer_sentiment = sentiment_df['sentiment_score'].mean() if not sentiment_df.empty else 0

    st.subheader(f"Average sentiment score for all customer responses: {overall_avg_customer_sentiment:.2f}/100")

    progress_value = int(overall_avg_customer_sentiment)
    st.progress(progress_value)

    st.dataframe(sentiment_df)

    # Aktualizacja wykresu liniowego
    st.session_state['ax2'].cla()  # Czyści oś wykresu, aby go zaktualizować
    x_values = range(1, len(st.session_state['all_customer_sentiments']) + 1)  # Liczby całkowite jako numery odpowiedzi
    st.session_state['ax2'].plot(x_values, st.session_state['all_customer_sentiments'],
                                 marker='o', color='skyblue', linestyle='-', linewidth=2)
    st.session_state['ax2'].set_xlabel('Number of Responses')
    st.session_state['ax2'].set_ylabel('Sentiment Score (1 = Negative, 100 = Positive)')
    st.session_state['ax2'].set_title('Sentiment Score Over Time')
    st.session_state['ax2'].set_ylim(0, 100)  # Zakres dla wyniku sentymentu od 0 do 100

    # Ustawienie etykiet na osi x tylko na liczby całkowite
    st.session_state['ax2'].set_xticks(x_values)

    st.pyplot(st.session_state['fig2'])


def call_real_time_analysis():
    st.logo('assets/logo_black.png', icon_image='assets/logo_magenta.png')



    # Whisper connection setup
    os.environ["OPENAI_API_KEY"] = "083d98847851416ea05e98396638d00a"
    os.environ[
        "OPENAI_API_BASE"] = "https://ai-us-2-team2.openai.azure.com/openai/deployments/whisper/audio/translations?api-version=2024-06-01"
    os.environ["OPENAI_API_VERSION"] = "2023-09-01-preview"

    # Create AzureOpenAI object
    azure_openai = AzureOpenAI(api_key=os.environ["OPENAI_API_KEY"],
                               azure_endpoint=os.environ["OPENAI_API_BASE"],
                               api_version=os.environ["OPENAI_API_VERSION"])

    deployment_id = "whisper"

    # Function to get responses from the AI
    def get_answer(conversation_history, name, character, language="english"):
        AZURE_API_KEY = "d2a8378ae86445be8873f41bf88ba72c"
        AZURE_OPENAI_ENDPOINT = "https://ai-challenge-team02.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

        headers = {
            "Content-Type": "application/json",
            "api-key": AZURE_API_KEY
        }

        # Create the message payload with the entire conversation history
        messages = [{"role": "system",
                     "content": f"You are an AI assistant and support representative at T-Mobile named {name}. You have a {character} demeanor. Answer in {language} Odpowiadaj W maksymalnie 2 zdaniach."}]

        for entry in conversation_history:
            messages.append(
                {"role": "user" if entry.startswith("User:") else "assistant", "content": entry.split(": ", 1)[1]})

        # Append user prompt to the messages
        if conversation_history:
            messages.append({"role": "user", "content": conversation_history[-1].split(": ", 1)[1]})

        payload = {
            "messages": messages,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 800
        }

        try:
            response = requests.post(AZURE_OPENAI_ENDPOINT, headers=headers, json=payload)
            answer = response.json()["choices"][0]["message"]["content"].strip()

        except Exception as e:
            st.error(f"Error: {e}")
            return "Przepraszam, wystąpił błąd."

        return answer


    # Initialize PyAudio
    p = pyaudio.PyAudio()

    def rms(data):
        """Calculates the Root Mean Square (RMS) of the audio signal."""
        audio_data = np.frombuffer(data, dtype=np.int16)
        return np.sqrt(np.mean(audio_data ** 2))

    def record_audio(filename):
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        st.write("Recording...")

        frames = []
        silence_chunks = 0
        silence_chunk_limit = int(SILENCE_DURATION * RATE / CHUNK)

        while True:
            data = stream.read(CHUNK)
            frames.append(data)

            if rms(data) < SILENCE_THRESHOLD:
                silence_chunks += 1
            else:
                silence_chunks = 0

            if silence_chunks > silence_chunk_limit:
                st.write("Silence detected, stopping recording.")
                break

        stream.stop_stream()
        stream.close()

        # Save recording to WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

    def transcribe_audio(filename):
        with open(filename, "rb") as audio_file:
            result = azure_openai.audio.transcriptions.create(
                file=audio_file,
                model=deployment_id,
                language="pl"
                #prompt="Rob transkrypcje po polsku, jezeli uslyszysz w innym jezyku przetlumacz na Polski"
            )
            return result.text

    # Streamlit application
    st.title("T-Mobile assistant Chatbot with customer sentiment analysis")

    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        st.session_state.introduced = False

    if not st.session_state.introduced:
        answer = get_answer(conversation_history=[], name=NAME, character=CHARACTER, language="english")
        st.write(f"{NAME}: {answer}")
        st.session_state.conversation.append(f"{NAME}: {answer}")
        st.session_state.introduced = True
        read_answer(answer)



    if st.button("Rozpocznij nagrywanie"):
        filename = "temp_audio.wav"
        record_audio(filename)
        text = transcribe_audio(filename)


        if text:
            st.write(f"User: {text}")
            st.session_state.conversation.append(f"User: {text}")
            answer = get_answer(conversation_history=st.session_state.conversation, name=NAME, character=CHARACTER,
                                language="polski")
            st.write(f"{NAME}: {answer}")
            st.session_state.conversation.append(f"{NAME}: {answer}")
            thread = threading.Thread(target=read_answer, args=(answer,))
            thread.start()
            analyze_sentiment_real_time(text)

    if st.button("Zakończ rozmowę"):
        st.write("Rozmowa zakończona.")
        st.session_state.conversation = []
        st.session_state.introduced = False

    if st.session_state.conversation:
        st.write("Historia rozmowy:")
        for line in st.session_state.conversation:
            st.write(line)

    p.terminate()