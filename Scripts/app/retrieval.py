import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class Retrieval:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None

    def load_data(self):
        """Load the dataset for retrieval."""
        if os.path.exists(self.data_path):
            self.data = pd.read_csv(self.data_path)
            self.data['content'] = self.data['content'].fillna('')
        else:
            raise FileNotFoundError(f"Data file not found at {self.data_path}")

    def build_index(self):
        """Build the TF-IDF index for retrieval."""
        if self.data is not None:
            self.tfidf_matrix = self.vectorizer.fit_transform(self.data['content'])
        else:
            raise ValueError("Data not loaded. Call load_data() first.")

    def retrieve(self, query, top_k=5):
        """Retrieve the top_k most relevant documents for a given query."""
        if self.tfidf_matrix is None:
            raise ValueError("Index not built. Call build_index() first.")

        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        return self.data.iloc[top_indices]

# Example usage
# retrieval = Retrieval(data_path="/path/to/your/dataset.csv")
# retrieval.load_data()
# retrieval.build_index()
# results = retrieval.retrieve("example query")
# print(results)
