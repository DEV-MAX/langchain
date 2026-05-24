from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter, SpacyTextSplitter, NLTKTextSplitter, TextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from torch import embedding
from langchain_core.vectorstores import InMemoryVectorStore

data = """
"Generative AI, sometimes called gen AI, is artificial intelligence (AI) 

that can create original content such as text, images, video, audio or software code in response to a user’s prompt or request."
"Generative AI relies on sophisticated machine learning models called deep learning models algorithms that simulate the learning and decision-making processes of the human brain. These models work by identifying and encoding the patterns and relationships in huge amounts of data, and then using that information to understand users' natural language requests or questions and respond with relevant new content."
"AI has been a hot technology topic for the past decade, but generative AI, and specifically the arrival of ChatGPT in 2022, has thrust AI into worldwide headlines and launched an unprecedented surge of AI innovation and adoption. Generative AI offers enormous productivity benefits for individuals and organizations, and while it also presents very real challenges and risks, businesses are forging ahead, exploring how the technology can improve their internal workflows and enrich their products and services. According to research by the management consulting firm McKinsey, one third of organizations are already using generative AI regularly in at least one business function.¹ Industry analyst Gartner projects more than 80% of organizations will have deployed generative AI applications or used generative AI application programming interfaces (APIs) by 2026.2"
"""

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

in_memory_vector_store = InMemoryVectorStore(embeddings)

def main():
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=10, chunk_overlap=0
    )
    texts = text_splitter.split_text(data)
    for text in texts:
        embeddedText= embeddings.embed_query(text)
        print(embeddedText)
 

def store_on_vectorstore():
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    encoding_name="cl100k_base", chunk_size=25, chunk_overlap=0
    )
    texts = text_splitter.split_text(data)

    documents = text_splitter.create_documents(texts)
    in_memory_vector_store.add_documents(documents)
    search_result=in_memory_vector_store.similarity_search("What is generative AI?", k=2)
    for doc in search_result:
        print(doc.page_content)
    #print(search_result)
 
if __name__ == "__main__":
    store_on_vectorstore()