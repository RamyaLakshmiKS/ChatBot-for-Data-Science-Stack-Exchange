from retrieval import Retrieval
from generation import Generation

class Chatbot:
    def __init__(self, data_path, model_name="gpt-2"):
        """Initialize the chatbot with retrieval and generation components."""
        self.retrieval = Retrieval(data_path)
        self.generation = Generation(model_name)

    def setup(self):
        """Load data and build the retrieval index."""
        self.retrieval.load_data()
        self.retrieval.build_index()

    def get_response(self, query):
        """Generate a response for the given query."""
        # Retrieve relevant documents
        retrieved_docs = self.retrieval.retrieve(query)
        context = " ".join(retrieved_docs['content'].tolist())

        # Generate a response based on the context
        response = self.generation.generate_response(f"{query} Context: {context}")
        return response

# Example usage
# chatbot = Chatbot(data_path="/path/to/your/dataset.csv")
# chatbot.setup()
# response = chatbot.get_response("What is the best way to learn Python?")
# print(response)
