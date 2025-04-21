import os
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
import logging
from ingest_data import query_batch_indices

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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


def load_vector_store(input_path="./data/faiss_index"):
    logging.debug(f"Loading FAISS vector store from {input_path}")
    # Load FAISS vector store from disk
    vector_store = FAISS.load_local(input_path, GoogleGenerativeAIEmbeddings(model="models/embedding-001"))
    logging.debug("FAISS vector store loaded successfully")
    return vector_store


def initialize_chatbot():
    logging.debug("Initializing chatbot")

    logging.debug("Loading batch FAISS indices")
    # Use batch FAISS indices for retrieval
    def batch_retriever(query, top_k=3):
        return query_batch_indices(query, output_path="./data/faiss_index", top_k=top_k)

    logging.debug("Initializing Gemini LLM")
    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

    logging.debug("Creating history-aware retriever")
    # Create history-aware retriever
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
        llm, batch_retriever, contextualize_q_prompt
    )

    logging.debug("Creating retrieval chain")
    # Create retrieval chain
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    chatbot = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    logging.debug("Chatbot initialization complete")
    return chatbot


# Streamlit app
def main():
    logging.debug("Starting Streamlit app")
    st.title("Data Science Q&A Chatbot")
    st.markdown(
        "Ask questions about data science, machine learning, and related topics!"
    )

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

        # Get response from chatbot
        with st.chat_message("assistant"):
            logging.debug("Getting response from chatbot")
            response = st.session_state.chatbot({"question": prompt})
            logging.debug(f"Chatbot response: {response['answer']}")
            st.markdown(response["answer"])
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": response["answer"]}
            )


if __name__ == "__main__":
    main()
