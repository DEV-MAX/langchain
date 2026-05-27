
from importlib import metadata
import os

from langchain_tavily import TavilyCrawl, TavilyMap
from langchain_pinecone import PineconeVectorStore, embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from openrouterembedding import OpenRouterEmbedding

from langchain_core.documents import Document

def get_batches(embeddings: list[Document], batch_size: int = 50):

    batches=[]

    split_documents=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(embeddings)
    #for i in range(0, len(embeddings), batch_size):
        
    #batch_embeddings = embeddings[i:i + batch_size]
    #    batch_metadata = metadata[i:i + batch_size]
    #    batches.append((batch_embeddings, batch_metadata))
    
    

    return [split_documents[i:i + batch_size] for i in range(0, len(split_documents), batch_size)]
    

async def save_embedding_asbatches(embeddings: list[list[Document]], batch_size: int = 10):
    result = [await _get_embeddings(batch) for batch in embeddings]
    flat_result=[item for ml in result for item in ml]
    return flat_result



async def _get_embeddings(embeddings: list[Document]):
    vector_store = PineconeVectorStore(index_name="langchain-default-index", embedding=OpenRouterEmbedding(), 
                                       pinecone_api_key=os.environ.get("PINE_CONE_API_KEY"))
    result=await vector_store.aadd_documents(embeddings)
    return result 
