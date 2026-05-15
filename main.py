from urllib import response

from langchain_openrouter import ChatOpenRouter
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()


def load_information():
    # This function can be used to load any necessary information or data
    # For example, you could load a knowledge base, user data, etc.
    return """Tamilaga Vettri Kazhagam (TVK) president C. Joseph Vijay will be sworn in as the Chief Minister of Tamil Nadu on Sunday (May 10, 2026), putting an end to almost six decades of alternating rule by the State’s two dominant Dravidian parties—DMK and AIADMK. Tamil Nadu Governor Rajendra Vishwanath Arlekar on Saturday (May 9, 2026) appointed Mr. Vijay as Chief Minister-designate after he secured the support of 120 MLAs-elect in the 234-member House.

                Mr. Vijay will take the oath at Nehru Stadium in Chennai at 10 a.m. on Sunday (May 10, 2026). Nine TVK Ministers are expected to be sworn along with Mr. Vijay.

                The Governor has also directed Mr. Vijay to seek a vote of confidence in the Assembly on or before May 13, 2026. 

                The nail-biting political uncertainty that prevailed over the last few days ended on Saturday (May 9) evening, when the VCK and the IUML extended their support, taking the TVK’s tally to 120. Later, Mr. Vijay called on the Governor at Lok Bhavan in Chennai, and handed over letters of support from the Congress, CPI, CPI (M), VCK and IUML.."
                """

def main():
    print("Hello from langchain!")
    print("Loading information...")
    information = load_information()
    print(f"Information loaded :{information}")

    # Create a prompt template
    prompt_template = PromptTemplate.from_template(
        "Based on the following information, answer the question: {question}\n\nInformation:\n{information}"
    )   
    # Create a ChatOpenRouter instance
    chat_router = ChatOpenRouter(temperature=0.1,model="inclusionai/ring-2.6-1t:free")
    # Define a question to ask
    question = "letter of support provide by?"
    # Get the response from the ChatOpenRouter
    chain = prompt_template| chat_router
    response=chain.invoke(input={"question": question, "information": information})
    print(f"Response: {response.content}")

if __name__ == "__main__":
    main()
