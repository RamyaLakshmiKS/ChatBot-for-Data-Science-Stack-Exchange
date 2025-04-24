# Data Science Stack Exchange Chatbot

This application is an advanced chatbot designed to answer data science-related questions using a Retrieval-Augmented Generation (RAG) approach. It leverages both a document retrieval system (FAISS) and large language models (Google Gemini) to provide accurate, context-aware, and conversational responses. The system is tailored for Stack Exchange-style Q&A, with additional features for question quality assessment and improvement suggestions.

---

## Features

- **Retrieval-Augmented Generation (RAG):** Combines semantic search (FAISS) with generative LLMs for high-quality, contextually relevant answers.
- **Question Quality Assessment:** Uses a Gemini LLM to evaluate the quality of user questions and predict the likelihood of receiving an answer.
- **Question Category Classification:** Classifies questions into categories (e.g., Data Science, Machine Learning, Statistics) using a LightGBM model.
- **Improvement Suggestions:** Provides actionable suggestions to improve question quality based on retrieved context and model feedback.
- **Interactive Streamlit UI:** Modern, user-friendly interface for seamless Q&A interactions.
- **Custom Data Ingestion:** Easily ingest and index new datasets using the provided ingestion script.

---

## Directory Structure

```
Scripts/app/
│
├── chatbot.py           # Main Streamlit app and chatbot logic
├── ingest_data.py       # Data ingestion and FAISS index creation
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
├── __init__.py
│
├── data/
│   ├── cleaned_stack_exchange_data.csv
│   └── faiss_index/
│       └── final_index.faiss/
│           ├── index.faiss
│           └── index.pkl
│
└── model/
    └── lgbm_model.pkl   # Trained LightGBM model for classification and ranking
```

---

## Setup Instructions

### 1. Install Dependencies

Navigate to the `Scripts/app` directory and install the required packages:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Prepare your dataset in CSV format (e.g., `cleaned_stack_exchange_data.csv`). The expected columns include `Title`, `Body`, `Tags`, `Score`, `ViewCount`, and other Stack Exchange post metadata.

Place your cleaned dataset in the `Scripts/app/data/` directory.

### 3. Data Ingestion & Indexing

Build the FAISS index for efficient semantic retrieval:

```bash
python ingest_data.py
```

This will process the CSV and create a FAISS index in `data/faiss_index/final_index.faiss/`.

### 4. Model Preparation

Ensure the trained LightGBM model (`lgbm_model.pkl`) is present in the `model/` directory. This model is used for question classification and ranking.

### 5. Configure Google API Key

To enable Gemini LLM features, add your Google API key:

1. Create a `.streamlit` directory in your project root (if it doesn't exist):

    ```bash
    mkdir -p .streamlit
    ```

2. Create a `secrets.toml` file inside `.streamlit`:

    ```toml
    [google]
    GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
    ```

Replace `YOUR_GOOGLE_API_KEY` with your actual key from Google AI Studio.

### 6. Run the Chatbot

Start the Streamlit app:

```bash
streamlit run chatbot.py
```

Open the provided URL in your browser to interact with the chatbot.

---

## How It Works

- **User Input:** Enter a data science question in the chat interface.
- **Category & Quality:** The system classifies the question, predicts its quality, and provides suggestions for improvement.
- **Retrieval:** Relevant Stack Exchange posts are retrieved using FAISS and HuggingFace embeddings.
- **LLM Response:** Gemini LLM generates a conversational answer, leveraging both retrieved context and chat history.
- **Feedback:** The UI displays question category, quality score, improvement suggestions, and the LLM's answer.

---

## Customization & Extensibility

- **Dataset:** You can ingest any Stack Exchange-style dataset by updating the CSV and re-running `ingest_data.py`.
- **Model:** Retrain or replace the LightGBM model as needed for different classification or ranking tasks.
- **UI:** Modify `chatbot.py` to customize the Streamlit interface or add new features.

---

## Troubleshooting

- Ensure all dependencies are installed and the FAISS index is built before running the chatbot.
- For issues with Google API access, verify your API key in `.streamlit/secrets.toml`.
- For further customization, review and edit `chatbot.py` and `ingest_data.py`.

---

## Credits

- Built with [Streamlit](https://streamlit.io/), [LangChain](https://www.langchain.com/), [FAISS](https://faiss.ai/), [HuggingFace Transformers](https://huggingface.co/), and [Google Gemini](https://ai.google.dev/).
- Data sourced from [Kaggle](https://www.kaggle.com/datasets/aneeshtickoo/data-science-stack-exchange?select=metadata.txt).