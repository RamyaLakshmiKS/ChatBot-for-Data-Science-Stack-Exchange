import json
import os
import traceback
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEmbeddings
import streamlit as st
import logging
import torch
import joblib
import numpy as np
import time
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Replace hardcoded API key with Streamlit secrets
logging.debug("Setting Google API key from Streamlit secrets.")
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def load_vector_store(input_path="./data/faiss_index/final_index.faiss"):
    logging.debug(f"Loading FAISS vector store from {input_path}")
    # Use HuggingFaceEmbeddings for consistency
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    )
    vector_store = FAISS.load_local(
        input_path, embeddings, allow_dangerous_deserialization=True
    )
    logging.debug("FAISS vector store loaded successfully")
    return vector_store


# Suggest better questions based on vector store context
def suggest_better_questions(question, vector_store):
    logging.debug("Suggesting better questions based on vector store context")
    # Retrieve context from the vector store
    retrieved_docs = vector_store.similarity_search(question, k=3)
    suggestions = [
        f"Consider asking about: {doc.metadata.get('title', 'No title')}"
        for doc in retrieved_docs
    ]
    return "\n".join(suggestions)


# Add question improvement suggestions
def suggest_improvements(question_text, model):
    logging.debug("Generating improvement suggestions")
    features = extract_features_from_text(question_text)  # Placeholder for actual feature extraction logic
    import shap
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values([features])
    # Generate suggestions based on SHAP values (placeholder logic)
    suggestions = generate_suggestions_from_shap(shap_values, features)  # Placeholder for actual suggestion logic
    return suggestions

def generate_suggestions_from_shap(shap_values, features):
    """
    Generate suggestions for improving a question based on SHAP values.

    Args:
        shap_values (list): SHAP values for the features.
        features (list): The original feature values.

    Returns:
        str: Suggestions for improving the question.
    """
    logging.debug("Generating suggestions from SHAP values")
    suggestions = []

    # Example logic: Check which features have the most negative impact
    for i, shap_value in enumerate(shap_values[0]):
        if shap_value < -0.1:  # Threshold for significant negative impact
            if i == 0:
                suggestions.append("Consider making your question longer.")
            elif i == 1:
                suggestions.append("Add more details to your question.")
            elif i == 3:
                suggestions.append("Ensure your question is clear and ends with a question mark.")

    return "\n".join(suggestions)

def extract_features_from_doc(doc):
    """
    Extract features from a document for scoring with the LightGBM model.

    Args:
        doc (Document): A LangChain Document object.

    Returns:
        list: A list of features extracted from the document.
    """
    logging.debug("Extracting features from document")

    # Example feature extraction logic (replace with actual logic from model training):
    text = doc.page_content
    features = extract_features_from_text(text)  # Reuse the text-based feature extraction

    # Add metadata-based features (if applicable)
    features.append(doc.metadata.get("score", 0))  # Example: Score of the post
    features.append(doc.metadata.get("view_count", 0))  # Example: View count of the post

    return features

# Modify hybrid retrieval system
def retrieve_documents(query, vector_store, model):
    logging.debug("Retrieving documents with hybrid scoring")
    docs = vector_store.similarity_search(query, k=10)
    scored_docs = []
    for doc in docs:
        features = extract_features_from_doc(doc)  # Placeholder for actual feature extraction logic
        quality_score = model.predict_proba([features])[0][1]
        scored_docs.append((doc, quality_score))
    return sorted(scored_docs, key=lambda x: x[1], reverse=True)


def initialize_chatbot():
    logging.debug("Initializing chatbot")

    logging.debug("Loading final FAISS index")
    vector_store = load_vector_store()

    logging.debug("Initializing Gemini LLM")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

    logging.debug("Creating history-aware retriever")
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(), contextualize_q_prompt
    )

    logging.debug("Creating retrieval chain")
    qa_system_prompt = (
        "You are an assistant for answering questions from Stack Exchange website. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, try to infer it from the chat history "
        "or provide a helpful response based on the context. Keep the answer "
        "concise and DO NOT add stuff from your memory unless it's absolutely necessary to address the question. "
        "Read the history of the conversation to understand the context of the question. "
        "\n\nContext: {context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    def chatbot(input_data):
        question = input_data["question"]
        chat_history = input_data.get("chat_history", [])

        formatted_input = {
            "input": question,
            "chat_history": chat_history,
        }
        response = retrieval_chain.invoke(formatted_input)
        return response

    logging.debug("Chatbot initialization complete")
    return chatbot


def extract_features_from_text(text):
    """
    Extract features from the input text for the LightGBM model.
    This function should replicate the feature engineering steps used during model training.

    Args:
        text (str): The input text (e.g., question content).

    Returns:
        list: A list of features extracted from the text.
    """
    logging.debug("Extracting features from text")

    # Example feature extraction logic (replace with actual logic from model training):
    features = []

    # Feature 1: Length of the text
    features.append(len(text))

    # Feature 2: Number of words in the text
    features.append(len(text.split()))

    # Feature 3: Number of unique words
    features.append(len(set(text.split())))

    # Feature 4: Presence of question mark
    features.append(1 if '?' in text else 0)

    # Feature 5: Average word length
    words = text.split()
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    features.append(avg_word_length)

    logging.debug(f"Extracted features: {features}")
    return features

# Replace LightGBM model with Gemini model for question quality prediction
# Update the chatbot initialization to use Gemini for quality prediction
logging.debug("Initializing Gemini LLM for question quality prediction")
quality_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)

# Define a function to predict question quality using Gemini

# Fix the input format for Gemini model and ensure proper response parsing
# Update the prompt for question quality prediction to handle abbreviations and well-structured questions
# Fix the response parsing logic to correctly extract the quality percentage
def predict_question_quality_with_gemini(question_text, llm):
    logging.debug("Predicting question quality using Gemini")
    prompt = [
        ("system", "You are a question quality evaluator. "
        "Provide a high quality score between 0 and 1. "
        "With 1 being awarded to a question closely related to questions that will be asked in data stack exchange and 0 being awarded to a question that is not related to data stack exchange. "
        "You will only give a JSON response with the following format: {\"score\": <float>}. Nothing else."
        "You are a question quality evaluator. Your task is to assess how well-structured, "
                "clear, and complete a given question is. Pay attention to the use of grammar, clarity, "
                "presence of abbreviations, and whether the question can be easily understood without additional context.\n\n"
                "Return a JSON object with the following keys:\n"
                "- 'score': A float score between 0.0 (poor quality) and 1.0 (excellent quality).\n"
                "- 'suggestions': A list of concise improvements if needed, such as expanding abbreviations, rephrasing for clarity, or adding missing context.\n\n"
                "Be fair and constructive. Only give a high score (e.g., > 0.8) if the question is clearly worded, unambiguous, and self-contained."),
        ("human", question_text)
    ]
    with st.spinner("Evaluating question quality..."):
        try:
            response = llm.invoke(prompt)
            logging.debug(f"Gemini response: {response}")
            # Ensure the response is parsed correctly to extract the quality score
            answer = json.loads(response.content)
            logging.debug(f"Parsed Gemini response: {answer}")
            return answer.get("score", 0.0)
        except ValueError:
            logging.error("Failed to parse quality score from Gemini response")
            return 0.0
        except Exception as e:
            logging.error(f"Unexpected error during quality prediction: {e}")
            traceback.print_exc()
            return 0.0

# Ensure asyncio event loop is properly initialized
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# Streamlit app
def main():
    # Set the page configuration with the desired tab header
    st.set_page_config(page_title="Ask, don't browse", page_icon=":robot_face:")

    logging.debug("Starting Streamlit app")
    st.title("Chat with Stack Exchange")
    st.markdown(
        "Stop wasting time searching for answers on the stack exchange website!"
    )
    st.markdown("Ask questions about data science, machine learning, and related topics!", unsafe_allow_html=True)

    # Initialize session state for chat history and chatbot
    if "messages" not in st.session_state:
        logging.debug("Initializing session state for messages")
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        logging.debug("Initializing session state for chatbot")
        st.session_state.chatbot = initialize_chatbot()

    # Remove references to the LightGBM model and ensure only Gemini is used for question quality prediction
    # Remove LightGBM model loading from session state
    if "lgbm_model" in st.session_state:
        del st.session_state["lgbm_model"]

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Add aesthetic features to the UI
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
            font-family: 'Arial', sans-serif;
        }
        .question-quality {
            font-size: 18px;
            font-weight: bold;
            color: #4CAF50;
        }
        .suggestions {
            font-size: 16px;
            color: #FF5722;
        }
        .loading {
            font-size: 20px;
            color: #2196F3;
            text-align: center;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Display a loading animation while the page is loading
    with st.spinner("Loading the chatbot interface..."):
        time.sleep(2)  # Simulate loading time

    # Fix asyncio event loop issue by ensuring a running loop
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Update the UI to display only the percentage and likelihood of getting an answer
    # Modify the Streamlit app to reflect the new requirement
    if prompt := st.chat_input("Ask a data science question!"):
        logging.debug(f"User input received: {prompt}")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]

        # Predict question quality using Gemini
        logging.debug("Evaluating question quality for UI display")
        quality_score = predict_question_quality_with_gemini(prompt, quality_llm)

        # Display question quality in the UI
        with st.chat_message("assistant"):
            quality_percentage = round(quality_score * 100, 2)
            likelihood = "likely to receive an answer" if quality_score > 0.5 else "unlikely to receive an answer"
            st.markdown(
                f"<div class='question-quality'>Question Quality: {quality_percentage}%\n\nYour question is {likelihood}.</div>",
                unsafe_allow_html=True
            )

        # Get response from chatbot
        with st.chat_message("assistant"):
            logging.debug("Getting response from chatbot")
            response = st.session_state.chatbot(
                {"question": prompt, "chat_history": chat_history}
            )
            logging.debug(f"Chatbot response: {response['answer']}")
            st.markdown(response["answer"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )

    # Disable Streamlit's file watcher for PyTorch modules
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    import sys

    if "torch" in sys.modules:
        sys.modules["torch"].__path__ = []


if __name__ == "__main__":
    main()