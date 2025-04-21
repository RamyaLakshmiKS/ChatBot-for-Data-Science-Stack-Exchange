import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging
from tqdm import tqdm
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_process_data(csv_path="./data/cleaned_stack_exchange_data.csv", batch_size=1000):
    logging.info(f"Loading data from {csv_path}")
    # Load CSV in chunks to reduce memory usage
    df_chunks = pd.read_csv(csv_path, chunksize=batch_size)

    documents = []
    for chunk in tqdm(df_chunks, desc="Processing data chunks"):
        logging.info("Processing a chunk of data")
        # Filter for question posts (PostTypeId == 1)
        questions_df = chunk[chunk["PostTypeId"] == 1].copy()

        # Combine relevant fields into a single text field
        questions_df["content"] = (
            "Title: " + questions_df["Title"].fillna("") + "\n" +
            "Body: " + questions_df["Body"].fillna("") + "\n" +
            "Tags: " + questions_df["Tags"].fillna("")
        )

        # Create LangChain Documents
        chunk_docs = [
            Document(
                page_content=row["content"],
                metadata={
                    "id": row["Id"],
                    "title": row["Title"],
                    "tags": row["Tags"],
                    "score": row["Score"],
                    "view_count": row["ViewCount"],
                }
            )
            for _, row in questions_df.iterrows()
        ]
        documents.extend(chunk_docs)

    logging.info("Splitting documents into chunks")
    # Use a larger chunk size to reduce the number of embeddings
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)

    logging.info(f"Processed {len(texts)} document chunks")
    return texts

def create_and_save_vector_store(texts, output_path="./data/faiss_index", batch_size=32):
    logging.info("Loading SentenceTransformer model 'all-mpnet-base-v2'")
    # Use HuggingFaceEmbeddings for compatibility with LangChain
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    logging.info("Creating FAISS vector store")
    # Create FAISS vector store with all documents at once
    vector_store = FAISS.from_documents(texts, embeddings)

    logging.info(f"Saving FAISS vector store to {output_path}")
    # Save FAISS index
    vector_store.save_local(output_path)

    logging.info("Vector store creation and saving complete")
    return vector_store

if __name__ == "__main__":
    logging.info("Starting data ingestion process")
    texts = load_and_process_data()
    create_and_save_vector_store(texts)
    logging.info("Data ingestion process complete")