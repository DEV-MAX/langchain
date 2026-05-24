from langchain_core.embeddings import Embeddings
from openrouter import OpenRouter
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file
import os

class OpenRouterEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.configuration = {
            "api_key": os.getenv("OPENROUTER_API_KEY")
        }
        # Initialize the embedding model here (e.g., load from Hugging Face)
      
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        with OpenRouter(api_key=self.configuration["api_key"]) as client:
            # Implement the logic to convert text to embeddings using the model
            # For example, you could use a pre-trained transformer model to generate embeddings
            response = client.embeddings.generate(model=self.model_name, input=texts)
            return [data.embedding for data in response.data]
        
    def embed_query(self, text: str) -> list:
        return self.embed_documents([text])[0]  # Embed the query as a single document and return the first embedding


if __name__ == "__main__":
    model_name = "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    embedding_model = OpenRouterEmbeddings(model_name)
    text = "This is a sample text to generate embeddings."
    embeddings = embedding_model.embed_query(text)
    print(embeddings)