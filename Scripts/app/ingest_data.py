import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
import logging
from tqdm import tqdm

import torch
from multiprocessing import Pool, cpu_count
import os

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

def process_batch(args):
    batch, embeddings, output_path, batch_index = args
    logging.info(f"Processing batch {batch_index}")
    vector_store = FAISS.from_documents(batch, embeddings)
    batch_path = os.path.join(output_path, f"batch_{batch_index}.faiss")
    vector_store.save_local(batch_path)
    return batch_path

def create_and_save_vector_store(texts, output_path="./data/faiss_index", batch_size=32):
    logging.info("Loading SentenceTransformer model 'all-mpnet-base-v2'")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    logging.info("Creating FAISS vector store in parallel")
    os.makedirs(output_path, exist_ok=True)

    # Prepare arguments for parallel processing
    args = [
        (texts[i:i + batch_size], embeddings, output_path, i // batch_size)
        for i in range(0, len(texts), batch_size)
    ]

    with Pool(cpu_count()) as pool:
        with tqdm(total=len(args), desc="Processing batches") as pbar:
            for _ in pool.imap_unordered(process_batch, args):
                pbar.update(1)

    logging.info("Merging FAISS vector store batches")
    # Load and merge all batch files
    vector_store = None
    for batch_path in tqdm([os.path.join(output_path, f"batch_{i}.faiss") for i in range(len(args))], desc="Merging batches"):
        if vector_store is None:
            vector_store = FAISS.load_local(batch_path, embeddings, allow_dangerous_deserialization=True)
        else:
            batch_vector_store = FAISS.load_local(batch_path, embeddings, allow_dangerous_deserialization=True)
            vector_store.merge_from(batch_vector_store)

    final_path = os.path.join(output_path, "final_index.faiss")
    vector_store.save_local(final_path)

    logging.info(f"Final FAISS vector store saved to {final_path}")
    return vector_store

def query_batch_indices(query, output_path="./data/faiss_index", top_k=5):
    logging.info("Loading batch FAISS indices for querying")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
    )

    batch_files = [
        os.path.join(output_path, f) for f in os.listdir(output_path) if f.startswith("batch_") and f.endswith(".faiss")
    ]

    results = []
    for batch_file in tqdm(batch_files, desc="Querying batches"):
        vector_store = FAISS.load_local(batch_file, embeddings, allow_dangerous_deserialization=True)
        batch_results = vector_store.similarity_search(query, k=top_k)
        results.extend(batch_results)

    # Sort results by score (if available) or relevance
    results = sorted(results, key=lambda x: x.metadata.get("score", 0), reverse=True)

    return results[:top_k]

if __name__ == "__main__":
    logging.info("Starting data ingestion process")
    texts = load_and_process_data()
    create_and_save_vector_store(texts)
    logging.info("Data ingestion process complete")