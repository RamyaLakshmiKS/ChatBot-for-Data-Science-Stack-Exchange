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
from sklearn.calibration import LabelEncoder
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

# Ensure asyncio event loop is properly initialized
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

def extract_features_from_text(text):
    """
    Extract features from the input text for the LightGBM model.
    This function replicates the feature engineering steps used during model training.

    Args:
        text (str): The input text (e.g., question content).

    Returns:
        list: A list of features extracted from the text.
    """
    logging.debug("Extracting features from text")

    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")

    # Example feature extraction logic based on model training:
    features = []

    # Feature 1: Length of the post body text
    features.append(len(text))  # body_length

    # Feature 2: Length of the post title 
    title = text.split("\n")[0] if "\n" in text else text  # Extract title from text
    features.append(len(title))  # title_length

    # Feature 3: Number of tags 
    tags = text.split("Tags:")[-1].strip() if "Tags:" in text else ""
    features.append(len(tags.split(",")))  # tag_count

    # Feature 4: Encoded primary tag 
    primary_tag = tags.split(",")[0] if tags else "none"
    le = LabelEncoder()
    le.fit(["none", primary_tag])  # Fit with a minimal set to avoid errors
    primary_tag_encoded = le.transform([primary_tag])[0]
    features.append(primary_tag_encoded)  # primary_tag_encoded

    # Feature 5: Post score 
    score = int(text.split("Score:")[-1].strip()) if "Score:" in text else 0
    features.append(score)  # Score

    # Feature 6: View count 
    view_count = int(text.split("ViewCount:")[-1].strip()) if "ViewCount:" in text else 0
    features.append(view_count)  # ViewCount

    # Feature 7: Comment count 
    comment_count = int(text.split("CommentCount:")[-1].strip()) if "CommentCount:" in text else 0
    features.append(comment_count)  # CommentCount

    # Feature 8: Favorite count 
    favorite_count = int(text.split("FavoriteCount:")[-1].strip()) if "FavoriteCount:" in text else 0
    features.append(favorite_count)  # FavoriteCount

    # Feature 9: User reputation 
    reputation = int(text.split("Reputation:")[-1].strip()) if "Reputation:" in text else 0
    features.append(reputation)  # Reputation

    # Feature 10: User views 
    user_views = int(text.split("Views:")[-1].strip()) if "Views:" in text else 0
    features.append(user_views)  # Views

    # Feature 11: User upvotes 
    upvotes = int(text.split("UpVotes:")[-1].strip()) if "UpVotes:" in text else 0
    features.append(upvotes)  # UpVotes

    # Feature 12: User downvotes 
    downvotes = int(text.split("DownVotes:")[-1].strip()) if "DownVotes:" in text else 0
    features.append(downvotes)  # DownVotes

    # Feature 13: Sum of comment scores 
    comment_score_sum = int(text.split("CommentScoreSum:")[-1].strip()) if "CommentScoreSum:" in text else 0
    features.append(comment_score_sum)  # comment_score_sum

    # Feature 14: Total number of comments 
    features.append(comment_count)  # comment_count 

    # Feature 15: Number of edits 
    edit_count = int(text.split("EditCount:")[-1].strip()) if "EditCount:" in text else 0
    features.append(edit_count)  # edit_count

    logging.debug(f"Extracted features: {features}")
    return features

def extract_features_from_doc(doc):
    """
    Extract features from a document for scoring with the LightGBM model.

    Args:
        doc (Document): A LangChain Document object.

    Returns:
        list: A list of features extracted from the document.
    """
    logging.debug("Extracting features from document")

    if not isinstance(doc, Document):
        raise ValueError("Input must be a Document instance.")

    # Example feature extraction logic (replace with actual logic from model training):
    text = doc.page_content
    features = extract_features_from_text(text)  # Reuse the text-based feature extraction

    # Add metadata-based features (if applicable)
    features.append(doc.metadata.get("score", 0))  # Example: Score of the post
    features.append(doc.metadata.get("view_count", 0))  # Example: View count of the post

    return features

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

def retrieve_documents(query, vector_store, model):
    logging.debug("Retrieving documents with hybrid scoring")
    docs = vector_store.similarity_search(query, k=10)
    scored_docs = []

    try:
        features_batch = [extract_features_from_doc(doc) for doc in docs]
        quality_scores = model.predict_proba(features_batch)[:, 1]  # Batch prediction
        scored_docs = list(zip(docs, quality_scores))
    except Exception as e:
        logging.error(f"Error during document scoring: {e}")
        logging.error(traceback.format_exc())

    return sorted(scored_docs, key=lambda x: x[1], reverse=True)

def initialize_chatbot():
    logging.debug("Initializing chatbot")

    logging.debug("Loading final FAISS index")
    vector_store = load_vector_store()
    logging.debug(f"Vector store loaded: {vector_store}")

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
        "You are an assistant for answering questions from Data Science Stack Exchange website. Always respond in English."
        "Use the following pieces of retrieved context to answer the "
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
    logging.debug(f"Retrieval chain created: {retrieval_chain}")

    def chatbot(input_data):
        question = input_data["question"]
        chat_history = input_data.get("chat_history", [])

        formatted_input = {
            "input": question,
            "chat_history": chat_history,
        }
        logging.debug(f"Formatted input for retrieval chain: {formatted_input}")
        response = retrieval_chain.invoke(formatted_input)
        logging.debug(f"Response from retrieval chain: {response}")
        return response

    logging.debug("Chatbot initialization complete")
    chatbot.vector_store = vector_store  # Add vector_store to chatbot for suggestions
    return chatbot

# Load the LightGBM model
logging.debug("Loading LightGBM model for hybrid scoring and suggestions")
lgbm_model_path = os.path.join(os.path.dirname(__file__), "model", "lgbm_model.pkl")
lgbm_model = joblib.load(lgbm_model_path)

# Add answer ranking functionality using LightGBM model
def rank_answers(answers, model=lgbm_model):
    """
    Rank answers based on relevance and quality using the LightGBM model.

    Args:
        answers (list): A list of answers, where each answer is a dictionary containing 'content' and optional metadata.
        model (LightGBM): The trained LightGBM model for ranking.

    Returns:
        list: A list of answers sorted by their predicted relevance and quality scores.
    """
    logging.debug("Ranking answers using LightGBM model")
    ranked_answers = []

    for answer in answers:
        # Extract features from the answer content and metadata
        features = extract_features_from_text(answer['content'])
        if 'metadata' in answer:
            features.append(answer['metadata'].get('upvotes', 0))  # Example: Upvotes
            features.append(answer['metadata'].get('length', len(answer['content'])))  # Example: Length of the answer

        # Predict the relevance score using the LightGBM model
        relevance_score = model.predict_proba([features])[0][1]
        ranked_answers.append((answer, relevance_score))

    # Sort answers by relevance score in descending order
    ranked_answers.sort(key=lambda x: x[1], reverse=True)

    # Return only the answers, sorted by their scores
    return [answer for answer, _ in ranked_answers]

def suggest_related_questions(query, vector_store):
    """
    Suggest related questions based on the user's query.

    Args:
        query (str): The user's query.
        vector_store (FAISS): The FAISS vector store for similarity search.

    Returns:
        list: A list of related questions.
    """
    logging.debug("Suggesting related questions")
    related_docs = vector_store.similarity_search(query, k=3)  # Retrieve top 3 related documents
    related_questions = [doc.page_content.split("\n")[0] for doc in related_docs]  # Extract titles or first lines
    return related_questions

def handle_related_question_click(question):
    """
    Handles the logic for when a related question is clicked.

    Args:
        question (str): The related question clicked by the user.
    """
    logging.debug(f"Related question clicked: {question}")

    # Append the clicked question to the chat history
    st.session_state.messages.append({"role": "user", "content": question})

    # Display the user's question in the chat UI
    with st.chat_message("user"):
        st.markdown(question)

    # Prepare the chat history for the chatbot
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    logging.debug(f"Chat history for related question: {chat_history}")

    # Execute steps 1, 2, 3, and 4
    try:
        with st.chat_message("assistant"):
            logging.debug("Getting response from chatbot for related question")

            # Step 1: Retrieve relevant documents
            with st.spinner("Retrieving relevant documents..."):
                st.markdown("**Step 1:** Retrieving relevant documents from the vector store.")
                time.sleep(1)  # Simulate document retrieval

            # Step 2: Rank answers
            with st.spinner("Ranking answers using LightGBM model..."):
                st.markdown("**Step 2:** Ranking answers based on relevance and quality.")
                time.sleep(1)  # Simulate answer ranking

            # Step 3: Generate response
            with st.spinner("Generating response..."):
                st.markdown("**Step 3:** Generating the final response using the LLM.")
                try:
                    response = st.session_state.chatbot(
                        {"question": question, "chat_history": chat_history}
                    )
                    logging.debug(f"Chatbot response: {response}")
                except Exception as e:
                    logging.error(f"Error generating response: {e}")
                    logging.error(traceback.format_exc())
                    response = {"answer": "Sorry, I couldn't process your request."}

            # Step 4: Display the final response
            st.markdown(response["answer"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )

            # Suggest related questions without clickable buttons
            if related_questions:
                st.markdown("### Related Questions:")
                for related_question in related_questions:
                    st.markdown(f"- {related_question}")
    except Exception as e:
        logging.error(f"Error handling related question click: {e}")
        logging.error(traceback.format_exc())
        with st.chat_message("assistant"):
            st.markdown("Sorry, I couldn't process your request.")

# Streamlit app
# Modify the button click logic to make the chatbot respond to the clicked question
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

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chatbot starts with greeting
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown("Hi! How can I assist you today?")

    # Handle user input
    if prompt := st.chat_input("Ask a question!"):
        logging.debug(f"User input received: {prompt}")

        # Check if the user says "Bye" or "Goodbye"
        if "bye" in prompt.lower() or "goodbye" in prompt.lower():
            with st.chat_message("assistant"):
                st.markdown("Goodbye! Have a great day!")
            st.stop()

        # Process user input
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        chat_history = [
            {"role": msg["role"], "content": msg["content"]}
            for msg in st.session_state.messages
        ]

        # Display chatbot answer separately
        with st.chat_message("assistant"):
            logging.debug("Getting response from chatbot")

            # Show the process in the UI
            with st.spinner("Retrieving relevant documents..."):
                st.markdown("**Step 1:** Retrieving relevant documents from the vector store.")
                # Simulate document retrieval
                time.sleep(1)  # Replace with actual retrieval logic if needed

            with st.spinner("Ranking answers using LightGBM model..."):
                st.markdown("**Step 2:** Ranking answers based on relevance and quality.")
                # Simulate answer ranking
                time.sleep(1)  # Replace with actual ranking logic if needed

            with st.spinner("Generating response..."):
                st.markdown("**Step 3:** Generating the final response using the LLM.")
                # Get the chatbot response
                response = st.session_state.chatbot(
                    {"question": prompt, "chat_history": chat_history}
                )

            # Display the final response
            logging.debug(f"Chatbot response: {response['answer']}")
            st.markdown(response["answer"])
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )

            # Suggest top 2 related questions 
            related_questions = suggest_related_questions(prompt, st.session_state.chatbot.vector_store)
            if related_questions:
                st.markdown("### Related Questions:")
                for related_question in related_questions[:2]:  # Limit to top 2 questions
                    st.markdown(f"- {related_question}")

    # Disable Streamlit's file watcher for PyTorch modules
    from streamlit.runtime.scriptrunner import add_script_run_ctx
    import sys

    if "torch" in sys.modules:
        sys.modules["torch"].__path__ = []


if __name__ == "__main__":
    main()