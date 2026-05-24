import os
os.environ["USER_AGENT"] = "my-langchain-app/gs/0.1.0"
from langchain_community.document_loaders import WebBaseLoader

URL="https://www.ibm.com/think/topics/generative-ai"
webloader=WebBaseLoader(URL)
documents=webloader.load()




def main():
    for doc in documents:
        print(doc.page_content)


if __name__ == "__main__":
    main()
