from langchain_pinecone import PineconeVectorStore
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage
from dotenv import load_dotenv
from langchain_core.documents import Document
import OpenRouterEmbeddings
load_dotenv()  # Load environment variables from .env file  
import os
from langchain_community.document_loaders import PyPDFLoader

index_name = "langchain-test-index"  # change if desired



def store_embeddings(documents:list[Document]):
    # Initialize PineCone client
    pinecone = PineconeVectorStore(pinecone_api_key=os.getenv("PINE_CONE_API_KEY"), 
                                    index_name="langchain-default", embedding=OpenRouterEmbeddings.OpenRouterEmbeddings("nvidia/llama-nemotron-embed-vl-1b-v2:free"))

    # Implement logic to store embeddings in PineCone

    pinecone.add_documents(documents=documents)


def retrieve_similar_documents(query, top_k=5) -> list[Document]:
    # Initialize PineCone client
    pinecone = PineconeVectorStore(pinecone_api_key=os.getenv("PINE_CONE_API_KEY"), 
                                    index_name="langchain-default", embedding=OpenRouterEmbeddings.OpenRouterEmbeddings("nvidia/llama-nemotron-embed-vl-1b-v2:free"))
    
    retriever=pinecone.as_retriever(search_kwargs={"k": top_k})
    result=retriever.invoke(input=query)
    return result
    # Implement logic to retrieve similar documents from PineCone based on the query
    # You can use the embedding model to convert the query into an embedding and then use PineCone's similarity search to find the most relevant documents
    # For example, you could embed the query and then use PineCone's search method to retrieve the top_k most similar documents based on the embeddings

def load_pdf(url):
    # Implement logic to load PDF and convert it to a list of Document objects
    # You can use libraries like PyPDF2 or pdfplumber to extract text from the PDF
    # For example, you could read the PDF, extract text, and create Document objects with the extracted text
    loader = PyPDFLoader(url)
    documents = loader.load()
    return documents
    
def get_content(documents:list[Document]):
    return [doc.page_content for doc in documents]


def ask_llm(question,context):
    # Implement logic to ask the LLM a question based on the retrieved documents
    # You can use the retrieved documents as context for the LLM to generate a response to the question
    # For example, you could concatenate the content of the retrieved documents and use it as input to the LLM along with the question to generate a response
    llm=ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free")
    prompt=[SystemMessage(content="You are a helpful assistant that answers questions based on the provided context. " \
    "Please use the retrieved documents to answer the question as accurately as possible. Do not make up information that is not present in the documents. If the answer is not present in the documents, please say 'I don't know'."),
            HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")]
    
    response=llm.invoke(input=prompt)
    return response.content

def join_content(documents:list[Document]):
    return "\n".join([doc.page_content for doc in documents])


def main():
    print("Hello from langchain!")
    url = "https://www.itic.org/documents/artificial-intelligence/ITI_AgenticAI_Final.pdf"
    #documents = load_pdf(url)
    #store_embeddings(documents) 
    response = retrieve_similar_documents("Foster transparency in the agentic AI value chain", top_k=1)
    content=join_content(response)
    response=ask_llm("What are the key points related to fostering transparency in the agentic AI value chain?", content)
    print(response)

if __name__ == "__main__":
    main()
