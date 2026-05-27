from langchain.embeddings import Embeddings
from openrouter import OpenRouter
from openrouter.operations import  CreateEmbeddingsResponse
from dotenv import load_dotenv
import os   
load_dotenv()

class OpenRouterEmbedding(Embeddings):

    model="text-embedding-3-small"
   
    def __init__(self, model: str=None, api_key: str=None):
        """Initialize with model and API key."""
        self.model = model or self.model    
        self.api_key = os.environ.get("OPENROUTER_API_KEY", api_key)
        self.client = OpenRouter(api_key=self.api_key)




    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed search docs.

        Args:
            texts: List of text to embed.

        Returns:
            List of embeddings.
        """

        try:             
            with self.client as client:
                return client.embeddings.generate(input=texts, model=self.model)
        except Exception as e:            
            print(f"Error embedding documents: {e}")
            return []
        
        
    

    def embed_query(self, text: str) -> list[float]:
        """Embed query text.

        Args:
            text: Text to embed.

        Returns:
            Embedding.
        """
        try:
            with self.client as client:
                response:CreateEmbeddingsResponse = client.embeddings.generate(input=[text], model=self.model)
                return [data.embedding for data in response.data]
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []
            

    
        
if __name__ == "__main__":
    embedding = OpenRouterEmbedding(model="nvidia/llama-nemotron-embed-vl-1b-v2:free")
    print(embedding.embed_query("What is the capital of France?"))