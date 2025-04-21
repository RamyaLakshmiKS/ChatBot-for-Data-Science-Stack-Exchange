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

# Replace hardcoded API key with Streamlit secrets
# Set your Google API key from Streamlit secrets
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

def load_and_process_data(csv_path="data_science_stack_exchange.csv"):
    # Load CSV
    df = pd.read_csv(csv_path)
    
    # Filter for question posts (PostTypeId == 1)
    questions_df = df[df['PostTypeId'] == 1].copy()
    
    # Combine relevant fields into a single text field
    questions_df['content'] = (
        "Title: " + questions_df['Title'].fillna("") + "\n" +
        "Body: " + questions_df['Body'].fillna("") + "\n" +
        "Tags: " + questions_df['Tags'].fillna("")
    )
    
    # Create LangChain Documents
    documents = [
        Document(
            page_content=row['content'],
            metadata={
                'id': row['Id'],
                'title': row['Title'],
                'tags': row['Tags'],
                'score': row['Score'],
                'view_count': row['ViewCount']
            }
        )
        for _, row in questions_df.iterrows()
    ]
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    texts = text_splitter.split_documents(documents)
    
    return texts

def create_vector_store(texts):
    # Create embeddings using Gemini
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Create FAISS vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    return vector_store

def initialize_chatbot():
    # Load and process data
    texts = load_and_process_data()

    # Create vector store
    vector_store = create_vector_store(texts)

    # Initialize Gemini LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.7)

    # Create history-aware retriever
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, vector_store.as_retriever(search_kwargs={"k": 3}), contextualize_q_prompt
    )

    # Create retrieval chain
    qa_system_prompt = (
        "You are an assistant for question-answering tasks. Use "
        "the following pieces of retrieved context to answer the "
        "question. If you don't know the answer, just say that you "
        "don't know. Use three sentences maximum and keep the answer "
        "concise."
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    chatbot = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return chatbot

# Streamlit app
def main():
    st.title("Data Science Q&A Chatbot")
    st.markdown("Ask questions about data science, machine learning, and related topics!")

    # Initialize session state for chat history and chatbot
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = initialize_chatbot()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Input prompt
    if prompt := st.chat_input("Ask a data science question!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from chatbot
        with st.chat_message("assistant"):
            response = st.session_state.chatbot({"question": prompt})
            st.markdown(response["answer"])
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})

if __name__ == "__main__":
    main()
