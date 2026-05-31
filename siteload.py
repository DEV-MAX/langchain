import streamlit as st
from main import getanswer

st.set_page_config(
    page_title="Streamlit App",
    page_icon=":sparkles:",
    layout="centered",
    initial_sidebar_state="expanded",
)


def initialize_messages():
    if "messages" not in st.session_state:
        st.session_state.messages = []


initialize_messages()
with st.sidebar:
    st.title("Know Langchain")
   
    newbutton=st.button("New")
    if newbutton:
        initialize_messages()


for message in st.session_state.messages:
    if message["role"] == "user":
        st.write(f"User: {message['content']}")
    else:
        st.write(f"Assistant: {message['content']}")


st.title("Welcome to the Streamlit App!")
st.write("Enter your prompt below and click the button to see the response.")


input=st.chat_input("Type your message here...", key="input")


if input:
    initialize_messages()
    usermessage = {"role": "user", "content": input}
    st.session_state.messages.append(usermessage)
    st.write(f"User: {input}")
    with st.spinner("AI is thinking..."):
        response = getanswer(input)  # Call the function from main.py
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.write(f"Assistant: {response}")



