# Chatbot Application

This chatbot application is designed to answer data science-related questions using a Retrieval-Augmented Generation (RAG) approach. It combines a retrieval system to fetch relevant documents and a generation system to create conversational responses. The chatbot is tailored to assist users with data science concepts, techniques, and best practices.

## Features

- Expertise in data science-related topics.
- Retrieval-Augmented Generation for accurate and context-aware responses.
- Easy-to-use interface for loading datasets and asking questions.
- Customizable to work with any dataset in CSV format.

## Setup Instructions

### 1. Install Dependencies

Ensure you have Python installed on your system. Install the required dependencies by running the following command in the `Scripts/app` directory:

```bash
pip install -r requirements.txt
```

### 2. Data Preparation

Prepare a dataset in CSV format with a column named `content`. The dataset should contain data science-related content. Place the dataset in the `Scripts/app/data/` directory or any other location of your choice.

### 3. Data Ingestion

Use the `ingest_data.py` script to preprocess the dataset and build the FAISS index for efficient retrieval. Run the following command:

```bash
python ingest_data.py --data_path data/cleaned_stack_exchange_data.csv
```

Replace `data/cleaned_stack_exchange_data.csv` with the path to your dataset if it is located elsewhere.

### 4. Running the Chatbot with Streamlit

To start the chatbot application using Streamlit, follow these steps:

1. Ensure the FAISS index is built and the dataset is prepared.
2. Run the Streamlit app using the following command:

```bash
streamlit run app.py
```

3. Open the provided URL in your browser to interact with the chatbot.

### Example Streamlit Code

Run the chatbot locally by running the following command in the app directory.

```
streamlit run chatbot.py
```

### 6. Adding Google API Key

To enable the chatbot to use Google AI services, you need to add your Google API key. Follow these steps:

1. Create a `.streamlit` directory in the root of your project if it doesn't already exist:

```bash
mkdir -p .streamlit
```

2. Create a `secrets.toml` file inside the `.streamlit` directory:

```bash
touch .streamlit/secrets.toml
```

3. Add your Google API key to the `secrets.toml` file in the following format:

```toml
[google]
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
```

Replace `YOUR_GOOGLE_API_KEY` with the actual API key obtained from Google AI Studio.

4. Save the file. The Streamlit app will automatically load the API key from this file when it runs.

## Additional Notes

- Ensure the FAISS index is built before running the chatbot.
- The chatbot is designed to work with datasets containing data science-related content. For other domains, you may need to retrain or fine-tune the model.
- For troubleshooting or further customization, refer to the `chatbot.py` and `ingest_data.py` scripts.