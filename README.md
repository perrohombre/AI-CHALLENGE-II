# AI Innovation Challenge II: Sentiment Analysis & Document Retrieval Application

This project, created as part of T-Mobile’s AI Innovation Challenge II, is an AI-driven application for real-time sentiment analysis of customer-sales interactions and efficient document search using Retrieval-Augmented Generation (RAG). The application leverages models like GPT-4o, Whisper, and Text-to-Speech (TTS) via Azure OpenAI to deliver real-time insights and document retrieval capabilities.

## Features

### Use Case I: Customer & Sales Sentiment Analysis
- **Real-Time Call Analysis**: Provides live sentiment analysis and real-time feedback during customer interactions.
- **Dashboard**: Allows selection of conversation count, analysis time estimation, and real-time visualization.
- **Customer Sentiment Insights**: Displays sentiment distribution and averages (histograms, tables).
- **Salesman Feedback**: Benchmarks salesman responses and offers customized feedback based on past interactions.

### Use Case II: RAG for Document Retrieval
- **PDF Processing**: Segments PDFs and generates embeddings to find the most relevant sections.
- **Contextual Response Generation**: Generates responses based on relevant document sections.

## Dataset Generation
A custom dataset generation script is included to create realistic customer-salesman conversations with varied tones and characteristics.

### Script Details
- **Customer Tones**: Randomly assigned based on a weighted distribution (e.g., "aggressive," "neutral," "friendly").
- **Salesman Profiles**: Features different salesman personas (e.g., "patient and understanding," "rude and impatient").
- **Output**: Generates a dataset of conversations saved in both JSON and CSV formats.

#### Running the Dataset Generation Script
The script `generate_dataset.py` uses Azure OpenAI to generate and save 1000+ conversations:
```python
# Example usage:
telecom_df = create_telecom_conversation_dataset_df(n_conversations=1100)
```

## Technologies
- **Frontend**: Streamlit
- **AI Models**: Azure OpenAI (GPT-4o, Whisper, TTS, Embeddings)
- **Data Visualization**: Matplotlib
- **Text Analysis**: Scikit-learn (Cosine Similarity)

## Setup Instructions

### Prerequisites
- Create an Azure OpenAI account and configure API keys.

### Clone the Repository
```bash
git clone https://github.com/perrohombre/AI-CHALLENGE-II
cd AI-CHALLENGE-II
```


### Run
```bash
streamlit run main.py
```

