# Chatbot Application

This chatbot application is built using a Retrieval-Augmented Generation (RAG) approach. It combines a retrieval system to fetch relevant documents and a generation system to create conversational responses.

## Requirements

Install the required dependencies using the following command:

```bash
pip install -r requirements.txt
```

## How to Use

### 1. Load Data

Ensure you have a dataset in CSV format with a column named `content`. Place the dataset in a known location and note its file path.

### 2. Initialize the Chatbot

Import the chatbot class and initialize it with the path to your dataset:

```python
from chatbot import Chatbot

# Initialize the chatbot
chatbot = Chatbot(data_path="/path/to/your/dataset.csv")
chatbot.setup()
```

### 3. Ask Questions

Use the `get_response` method to ask questions and get responses:

```python
response = chatbot.get_response("What is the best way to learn Python?")
print(response)
```

### Example

```python
from chatbot import Chatbot

# Initialize the chatbot
chatbot = Chatbot(data_path="/path/to/your/dataset.csv")
chatbot.setup()

# Ask a question
response = chatbot.get_response("What is the capital of France?")
print(response)
```
