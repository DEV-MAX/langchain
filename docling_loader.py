
from langchain_docling.loader import DoclingLoader
import os
os.environ["USER_AGENT"] = "my-langchain-app/gs/0.1"


documents=DoclingLoader("https://www.ibm.com/think/topics/generative-ai").load()

def main():
    for doc in documents:
        print(doc.page_content) 


if __name__ == "__main__":
    main()   