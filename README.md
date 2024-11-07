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

