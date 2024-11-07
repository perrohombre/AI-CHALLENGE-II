import random
import requests
import json
import pandas as pd
import time
from dotenv import load_dotenv
import os

# Ustawienia API
AZURE_COMPLETIONS_API_KEY = os.getenv('AZURE_COMPLETIONS_API_KEY')
AZURE_COMPLETIONS_OPENAI_ENDPOINT = os.getenv('AZURE_COMPLETIONS_OPENAI_ENDPOINT')

# Możliwe tony klienta z przypisaniem im wag dla rozkładu Gaussa
tones = ["aggressive", "angry", "neutral", "polite", "friendly"]
# Przypisanie wag dla Gaussa (większa waga dla neutralnych tonów)
weights = [0.05, 0.1, 0.7, 0.1, 0.05]  # Suma wag = 1, neutralne mają największą wagę

# Lista 10 salesmanów z unikalnymi ID i charakterami (w tym negatywne cechy)
salesmen = [
    {"id": 1, "name": "John", "character": "patient and understanding"},
    {"id": 2, "name": "Alice", "character": "quick and efficient"},
    {"id": 3, "name": "Bob", "character": "formal and professional"},
    {"id": 4, "name": "Carol", "character": "casual and friendly"},
    {"id": 5, "name": "David", "character": "impatient and rude"},
    {"id": 6, "name": "Eve", "character": "unhelpful and dismissive"},
    {"id": 7, "name": "Frank", "character": "straightforward and direct"},
    {"id": 8, "name": "Grace", "character": "soft-spoken and polite"},
    {"id": 9, "name": "Hank", "character": "distracted and uninterested"},
    {"id": 10, "name": "Ivy", "character": "energetic and enthusiastic"},
    {"id": 11, "name": "Jack", "character": "confused and uncertain"},
    {"id": 12, "name": "Kelly", "character": "pushy and aggressive"},
    {"id": 13, "name": "Leo", "character": "helpful and supportive"},
    {"id": 14, "name": "Mia", "character": "sarcastic and witty"},
    {"id": 15, "name": "Nick", "character": "perfect salesperson"},
    {"id": 16, "name": "Olivia", "character": "charming and persuasive"},
    {"id": 17, "name": "Peter", "character": "knowledgeable and informative"},
    {"id": 18, "name": "Quinn", "character": "extremly rude, hates customers"},
    {"id": 19, "name": "Rachel", "character": "confident and assertive"},
    {"id": 20, "name": "Sam", "character": "funny and entertaining"}
]

# Funkcja do generowania promptu
def generate_prompt(tone, salesman_character, industry="telecommunication", company="T-Mobile"):
    return f"""
You are an AI assistant. Create a realistic conversation between a customer and a salesman at a {company} {industry} company.
The conversation should be based on telecom services such as plans, pricing, or technical support.
The customer is in a {tone} mood.
The salesman has the following characteristics: {salesman_character}.

Return the conversation in strict JSON format, with the structure:

[
  {{ "role": "customer", "text": "<customer's sentence>" }},
  {{ "role": "salesman", "text": "<salesman's response>" }}
]

Make sure the response is well-formed JSON and does not contain any additional explanation.
"""

# Funkcja do wysyłania zapytania do Azure OpenAI
def get_conversation_from_openai(prompt):
    headers = {
        "Content-Type": "application/json",
        "api-key": AZURE_COMPLETIONS_API_KEY
    }
    
    payload = {
        "messages": [
            {"role": "system", "content": "You are a conversational assistant specialized in customer service in telecommunication."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 1000,
        "top_p": 0.95
    }
    
    try:
        response = requests.post(AZURE_COMPLETIONS_OPENAI_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Sprawdzenie poprawności odpowiedzi
        try:
            conversation = response.json()["choices"][0]["message"]["content"]
        except KeyError:
            print(f"Failed to get conversation from response: {response.json()}")
            return None
        
        # Próba parsowania odpowiedzi jako JSON
        try:
            conversation_json = json.loads(conversation)
            return conversation_json
        except json.JSONDecodeError:
            print(f"Failed to parse JSON response: {conversation}")
            return None
    except requests.RequestException as e:
        print(f"Error during OpenAI request: {e}")
        return None

# Funkcja do generowania datasetu 100 konwersacji i zapisywania w DataFrame
def create_telecom_conversation_dataset_df(n_conversations=100):
    dataset = []
    
    for i in range(n_conversations):
        # Losowy wybór tonu klienta na podstawie rozkładu wag
        customer_tone = random.choices(tones, weights=weights)[0]
        
        # Losowy wybór salesmana
        selected_salesman = random.choice(salesmen)
        
        # Generowanie promptu z uwzględnieniem charakteru salesmana
        prompt = generate_prompt(customer_tone, selected_salesman["character"])
        
        # Mierzenie czasu generacji rozmowy
        start_time = time.time()  # Start pomiaru czasu
        
        # Pobranie odpowiedzi z Azure OpenAI
        conversation = get_conversation_from_openai(prompt)
        
        end_time = time.time()  # Koniec pomiaru czasu
        generation_time = end_time - start_time  # Obliczenie czasu generacji

        # Wyświetlenie czasu generacji dla tej rozmowy
        print(f"Conversation {i+1} generated in {generation_time:.2f} seconds.")
        
        if conversation:
            dataset.append({
                "id": i + 1,
                "salesman_id": selected_salesman["id"],
                "salesman_name": selected_salesman["name"],
                "salesman_character": selected_salesman["character"],
                "conversation": conversation
            })
        else:
            print(f"Error parsing conversation {i+1}, skipping this one.")
    
    # Tworzenie DataFrame z datasetu
    df = pd.DataFrame(dataset)
    
    # Zapis do pliku JSON
    with open("telecom_conversations_dataset.json", "w", encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    
    # Zapis do pliku CSV
    df.to_csv("telecom_conversations_dataset.csv", index=False)
    
    print(f"Dataset of {len(dataset)} conversations generated and saved as 'telecom_conversations_dataset.json' and 'telecom_conversations_dataset.csv'.")
    
    return df

# Generowanie datasetu i zapis do DataFrame
telecom_df = create_telecom_conversation_dataset_df(n_conversations=1100)

# Wyświetlenie pierwszych 5 wierszy
print(telecom_df.head())

# Zapis DataFrame do pliku CSV
telecom_df.to_csv('telecom_1000.csv', index=False)