import os
from langchain.messages import HumanMessage, ToolMessage
from langchain_pinecone import PineconeVectorStore
from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langchain.agents import create_agent

from openrouterembedding import OpenRouterEmbedding


@tool(
    description="Takes the query and retrieves context from vector store",
    response_format="content_and_artifact",
)
def retrive_context(query: str):
    vector_store = PineconeVectorStore(
        index_name="langchain-default-index",
        embedding=OpenRouterEmbedding(),
        pinecone_api_key=os.environ.get("PINE_CONE_API_KEY"),
    )

    retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    observation = "\n\n".join(
        [
            f"Content: {doc.page_content}\nMetadata: {doc.metadata}"
            for doc in docs
        ]
    )

    return observation, docs


model = ChatOpenRouter(
    model="nvidia/nemotron-3-super-120b-a12b:free",
    temperature=0.1,
)

system_prompt = """You are a helpful assistant that answers questions based on the provided context.
Use the retrive_context tool to get relevant information from the vector store.
Provide concise and accurate answers based on the retrieved context.
If the context does not contain the answer, respond with "I don't know".
Always use the retrive_context tool before answering.
"""

agent = create_agent(
    model=model,
    tools=[retrive_context],
    system_prompt=system_prompt,
)


def ask_llm(query: str):
    result = agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    answer= result["messages"][-1].content


    documents= [item.artifact for item in result["messages"] if isinstance(item, ToolMessage) and hasattr(item,"artifact")]
    return {"answer": answer, "retrieved_docs": documents}

if __name__ == "__main__":
    answer = ask_llm("how to create an agent for model on openrouter?")
    print(answer.get("answer"))