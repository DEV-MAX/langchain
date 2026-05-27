
import asyncio
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from openrouterembedding import OpenRouterEmbedding
import crawl

import ingestion

def main():
    print("Hello from langchain!")


def  crawl_website(url: str) -> list[dict[str, str]]:
    response = crawl.crawl(url)
    
    return [Document(page_content=content["content"], metadata={"url": content["url"]}) for content in response]




if __name__ == "__main__":
    main()
    documents = crawl_website("https://python.langchain.com")
    batches=ingestion.get_batches(documents, batch_size=10)
    embedding = asyncio.run(ingestion.save_embedding_asbatches(batches))
    print(embedding)