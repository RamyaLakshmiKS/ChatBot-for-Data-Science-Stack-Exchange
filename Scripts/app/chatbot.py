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

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Replace hardcoded API key with Streamlit secrets
logging.debug("Setting Google API key from Streamlit secrets.")
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]


def load_and_process_data(
    csv_path="./data/cleaned_stack_exchange_data.csv",
):
    logging.debug(f"Loading data from {csv_path}")
    # Load CSV
    df = pd.read_csv(csv_path)

    logging.debug("Filtering for question posts (PostTypeId == 1)")
    # Filter for question posts (PostTypeId == 1)
    questions_df = df[df["PostTypeId"] == 1].copy()

    logging.debug("Combining relevant fields into a single text field")
    # Combine relevant fields into a single text field
    questions_df["content"] = (
        "Title: "
        + questions_df["Title"].fillna("")
        + "\n"
        + "Body: "
        + questions_df["Body"].fillna("")
        + "\n"
        + "Tags: "
        + questions_df["Tags"].fillna("")
    )

    logging.debug("Creating LangChain Documents")
    # Create LangChain Documents
    documents = [
        Document(
            page_content=row["content"],
            metadata={
                "id": row["Id"],
                "title": row["Title"],
                "tags": row["Tags"],
                "score": row["Score"],
                "view_count": row["ViewCount"],
            },
        )
        for _, row in questions_df.iterrows()
    ]

    logging.debug("Splitting documents into chunks")
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    logging.debug("Data processing complete")
    return texts


def create_vector_store(texts):
    logging.debug("Creating embeddings using Google Generative AI")
    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    logging.debug("Creating FAISS vector store")
    # Create FAISS vector store
    vector_store = FAISS.from_documents(texts, embeddings)

    logging.debug("Vector store creation complete")
    return vector_store


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


def initialize_chatbot():
    logging.debug("Initializing chatbot")

    logging.debug("Loading final FAISS index")
    # Load final FAISS index for retrieval
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
        "You are an assistant for answering questions from Stack Exchange wesbite. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Keep the answer"
        "concise and DO NOT add stuff from your memory unless it's absolutely necessary to address the question."
        "read the history of the conversation to understand the context of the question."
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
        formatted_input = {
            "input": input_data["question"],
            "chat_history": input_data.get("chat_history", []),  # Default to an empty list if not provided
        }
        response = retrieval_chain.invoke(formatted_input)

        # Log the retrieved context for debugging purposes
        retrieved_context = response.get("context", "No context retrieved")
        logging.debug(f"Retrieved context: {retrieved_context}")

        return response

    logging.debug("Chatbot initialization complete")
    return chatbot


# Streamlit app
def main():
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

    # Input prompt
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


if __name__ == "__main__":
    main()
