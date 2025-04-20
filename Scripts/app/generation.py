from transformers import pipeline

class Generation:
    def __init__(self, model_name="gpt-2"):
        """Initialize the generation model."""
        self.generator = pipeline("text-generation", model=model_name)

    def generate_response(self, prompt, max_length=50):
        """Generate a response based on the given prompt."""
        responses = self.generator(prompt, max_length=max_length, num_return_sequences=1)
        return responses[0]['generated_text']

# Example usage
# generation = Generation()
# response = generation.generate_response("What is the capital of France?")
# print(response)
