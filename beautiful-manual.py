from urllib import response

from bs4 import BeautifulSoup
import requests


heading={
    "USER_AGENT": "langchain-app/gs/0.1",
}



def main():
    response=requests.get("https://www.ibm.com/think/topics/generative-ai", headers=heading)
    beautifulsoup=BeautifulSoup(response.text, "html.parser")
    print(beautifulsoup.find("body").prettify())


    


if __name__ == "__main__":
    main();        