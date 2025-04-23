import os
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


# Load the LightGBM model during chatbot initialization
def load_lgbm_model(model_path=".\..\milestone_2\lgbm_model.pkl"):
    logging.debug(f"Loading LightGBM model from {model_path}")
    model = joblib.load(model_path)
    logging.debug("LightGBM model loaded successfully")
    return model


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


# Add predictive answer quality assessment
def evaluate_question_quality(question_text, model):
    logging.debug("Evaluating question quality")
    # Extract features from the question text (ensure feature extraction matches training)
    features = extract_features_from_text(question_text)  # Placeholder for actual feature extraction logic
    answer_probability = model.predict_proba([features])[0][1]
    return answer_probability

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
    # Load final FAISS index for retrieval
    vector_store = load_vector_store()

    logging.debug("Loading LightGBM model")
    lgbm_model = load_lgbm_model()

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

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(
        history_aware_retriever, question_answer_chain
    )

    # Ensure the chatbot is callable by directly using the retrieval chain
    def chatbot(input_data):
        question = input_data["question"]
        chat_history = input_data.get("chat_history", [])

        # Predict question quality
        logging.debug("Predicting question quality")
        quality_score = evaluate_question_quality(question, lgbm_model)

        if quality_score > 0.5:  # Threshold for good quality question
            formatted_input = {
                "input": question,
                "chat_history": chat_history,
            }
            response = retrieval_chain.invoke(formatted_input)
            return response
        else:
            # Suggest improvements if the question quality is low
            suggestions = suggest_improvements(question, lgbm_model)
            return {"answer": f"Your question might not receive an answer. Here are some suggestions:\n{suggestions}"}

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

    # Store the LightGBM model in session state during chatbot initialization
    if "lgbm_model" not in st.session_state:
        logging.debug("Loading LightGBM model into session state")
        st.session_state.lgbm_model = load_lgbm_model()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Modify the Streamlit app to display question quality
    if prompt := st.chat_input("Ask a data science question!"):
        logging.debug(f"User input received: {prompt}")
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Prepare chat history for the chatbot
        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]

        # Evaluate question quality
        logging.debug("Evaluating question quality for UI display")
        quality_score = evaluate_question_quality(prompt, st.session_state.lgbm_model)

        # Display question quality in the UI
        with st.chat_message("assistant"):
            if quality_score > 0.5:  # Threshold for good quality question
                quality_percentage = round(quality_score * 100, 2)
                st.markdown(
                    f"**Question Quality:** {quality_percentage}%\n\nYour question is likely to receive an answer."
                )
            else:
                st.markdown("**Question Quality:** Poor\n\nYour question might not receive an answer. Here are some suggestions:")
                suggestions = st.session_state.chatbot.suggest_improvements(prompt)
                st.markdown(suggestions)

        # Get response from chatbot
        with st.chat_message("assistant"):
            logging.debug("Getting response from chatbot")
            response = st.session_state.chatbot(
                {"question": prompt, "chat_history": chat_history}
            )
            logging.debug(f"Chatbot response: {response['answer']}")
            st.markdown(response["answer"])
            # Add assistant response to chat history
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
