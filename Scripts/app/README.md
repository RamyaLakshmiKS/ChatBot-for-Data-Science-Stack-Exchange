# Chatbot Application

This chatbot application is designed to specialize in answering data science-related questions. It uses a Retrieval-Augmented Generation (RAG) approach, combining a retrieval system to fetch relevant documents and a generation system to create conversational responses. The chatbot is tailored to assist users with data science concepts, techniques, and best practices.

## Features

- Expertise in data science-related topics.
- Retrieval-Augmented Generation for accurate and context-aware responses.
- Easy-to-use interface for loading datasets and asking questions.
- Customizable to work with any dataset in CSV format.

## Requirements

Ensure you have Python installed on your system. Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Load Data

Prepare a dataset in CSV format with a column named `content`. The dataset should contain data science-related content. Place the dataset in a known location and note its file path.

### 2. Initialize the Chatbot

Import the chatbot class and initialize it with the path to your dataset:

```python
from chatbot import Chatbot

# Initialize the chatbot
chatbot = Chatbot(data_path="/path/to/your/dataset.csv")
chatbot.setup()
```

### 3. Ask Questions

Use the `get_response` method to ask data science-related questions and get responses:

```python
response = chatbot.get_response("What is the difference between supervised and unsupervised learning?")
print(response)
```

### Example

```python
from chatbot import Chatbot

# Initialize the chatbot
chatbot = Chatbot(data_path="/path/to/your/dataset.csv")
chatbot.setup()

# Ask a question
response = chatbot.get_response("Can you explain the concept of overfitting in machine learning?")
print(response)
```