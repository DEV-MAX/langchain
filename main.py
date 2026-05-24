from urllib import response

from langchain_pinecone import PineconeVectorStore
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda, chain
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from operator import itemgetter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
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


def get_retriever(top_k=5) -> VectorStoreRetriever:
    # Initialize PineCone client
    pinecone = PineconeVectorStore(pinecone_api_key=os.getenv("PINE_CONE_API_KEY"), 
                                    index_name="langchain-default", embedding=OpenRouterEmbeddings.OpenRouterEmbeddings("nvidia/llama-nemotron-embed-vl-1b-v2:free"))
    
    retriever=pinecone.as_retriever(search_kwargs={"k": top_k})
    return retriever
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


def get_prompt(input): 
    prompt = [
        SystemMessage(
            content=(
                "You are a helpful assistant that answers questions based on the provided context. "
                "Please use the retrieved documents to answer the question as accurately as possible. "
                "Do not make up information that is not present in the documents. "
                "If the answer is not present in the documents, please say 'I don't know'."
            )
        ),
        HumanMessage(
            content=f"Context: {input['context']}\n\nQuestion: {input['question']}"
        )
    ]

    return prompt
   
       
llm=ChatOpenRouter(model="nvidia/nemotron-3-super-120b-a12b:free")



def join_content(documents:list[Document]):
    return "\n".join([doc.page_content for doc in documents])


def main():
    print("Hello from langchain!")
    url = "https://www.itic.org/documents/artificial-intelligence/ITI_AgenticAI_Final.pdf"
    #documents = load_pdf(url)
    #store_embeddings(documents) 
    retriever = get_retriever(top_k=5)
    query = "What are the main challenges of agentic AI?"
    chain = {
            "context": itemgetter("question")  | retriever | RunnableLambda(join_content),
            "question": itemgetter("question"),
            }|RunnableLambda(get_prompt) | llm | StrOutputParser()
    llm_response = chain.invoke({"question": "What are the main challenges of agentic AI?"})
    print(llm_response)

if __name__ == "__main__":
    main()
