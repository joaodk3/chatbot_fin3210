import streamlit as st
from openai import OpenAI
import pdfplumber
import re
from openai import OpenAIError
import os
import time
from functools import wraps
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.memory import ConversationBufferMemory

def rate_limit(max_per_minute):
    min_interval = 60.0 / max_per_minute
    last_called = [0.0]

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

units = {
    "Unit 1 - Introduction": os.path.join("assets", "Unit1_Intro.pdf"),
    'Unit 2 - Financial Statements': os.path.join("assets", "Unit2_FinancialStatements.pdf"),
    'Unit 3 - Time Value of Money': os.path.join("assets", "Unit3_TimeValueOfMoney_annotated.pdf"),
    'Unit 4 - Bonds': os.path.join("assets", "Unit4_BONDS_Annotated.pdf"),
    'Unit 5 - Stocks': os.path.join("assets", "Unit5_Stocks_SlideDeck_annotated.pdf"),
    'Unit 6 - Capital Budgeting': os.path.join("assets", "Capital_Budgeting.pdf")
}

gpt_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4o"]

if "current_model" not in st.session_state:
    st.session_state.current_model = gpt_models[0]

if "vector_stores" not in st.session_state:
    st.session_state.vector_stores = {}

@st.cache_data(show_spinner=False)
def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    text = re.sub(r'\n+', '\n', text)  
    text = re.sub(r'\s+', ' ', text)   
    return text

def create_vector_store(text, embeddings):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return FAISS.from_texts(chunks, embeddings)

def get_vector_store(option, api_key):
    if option not in st.session_state.vector_stores:
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        text = read_pdf(units[option])
        st.session_state.vector_stores[option] = create_vector_store(text, embeddings)
    return st.session_state.vector_stores[option]

def create_chain(vector_store, api_key, model_name):
    llm = ChatOpenAI(
        model_name=model_name,
        openai_api_key=api_key,
        temperature=0.7,
        streaming=True
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI teaching assistant for a finance course. Answer questions based on the following context. 
        If the question cannot be answered from the context, politely inform the student that the topic might be covered in a different unit.
        
        Context: {context}
        """),
        ("human", "{question}")
    ])

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3} 
    )

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain

def validate_api_key(api_key):
    client = OpenAI(api_key=api_key)
    try:
        client.models.list()
        return True, None
    except OpenAIError as e:
        if "invalid_api_key" in str(e).lower():
            return False, "Invalid API key. Please check and try again."
        else:
            return False, str(e)

@rate_limit(max_per_minute=60)
def get_teacher_response(chain, message):
    try:
        response = chain.stream(message)
        return response
    except OpenAIError as e:
        if "insufficient_quota" in str(e).lower():
            st.error("❌ You have insufficient credits to perform this operation.")
        else:
            st.error(f"❌ An error occurred: {str(e)}")
        return None
    
def faq_section():
    st.markdown("### Frequently Asked Questions (FAQ)")

    with st.expander("How Does the App Work?"):
        st.write("""This app allows you to ask questions about Managerial Finance topics. The chatbot is powered by OpenAI's GPT models, which read the selected unit's PDF content and provide answers based on it. Make sure to provide a valid API key to interact with the chatbot.""")

    with st.expander("How to get an OpenAI Key?"):
        st.write("""You can get an OpenAI API key by signing up at [OpenAI's API platform](https://platform.openai.com/signup). Once signed up, navigate to the API section in your account to generate an API key.""")

    with st.expander("What is an API Key?"):
        st.write("""An API key is a unique identifier used to authenticate requests made to the OpenAI API. It allows your app to securely interact with OpenAI's models.""")

    with st.expander("Is the API Key Free?"):
        st.write("""OpenAI offers limited free access to its API, but usage beyond certain limits requires a paid subscription. You can check their pricing on the [OpenAI pricing page](https://openai.com/pricing).""")

    with st.expander("How to keep track of your API credits usage?"):
        st.write("You can track your OpenAI API usage by visiting the [Usage Dashboard](https://platform.openai.com/account/usage).")


def main():
    st.title("🤓🧮 FIN3210 Chatbot")

    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(return_messages=True)

    with st.sidebar:
        st.image("https://asset4.applyesl.com/images/sch/0134800/04.jpg")
        api_key = st.text_input("🔑 Insert your OpenAI API key", type="password")
        new_model = st.selectbox("Select a GPT Model", gpt_models)
        
        if new_model != st.session_state.current_model:
            st.session_state.current_model = new_model
            st.session_state.messages = []
            st.session_state.memory.clear()
        
        units_with_default = {"Select a unit": ""} | units
        option = st.selectbox("📓 Select the Class Topic", list(units_with_default.keys()))
        if option != "Select a unit":
            st.write("You selected:", option)
        
        st.divider()
        faq_section()
        st.divider()
        
        if st.button("Clear Conversation", type="secondary"):
            st.session_state.messages = []
            st.session_state.memory.clear()

    if not api_key:
        st.error("🔒 Please enter your OpenAI API key to continue.")
        st.link_button("Get an OpenAI API Key", "https://platform.openai.com/account/api-keys", type='secondary')
        return

    is_valid, error_message = validate_api_key(api_key)
    if not is_valid:
        st.error(f"❌ {error_message}")
        return

    if option == "Select a unit":
        st.error("📚 Please select a class unit to continue.")
        return

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the selected unit"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        vector_store = get_vector_store(option, api_key)
        chain = create_chain(vector_store, api_key, st.session_state.current_model)
        
        response_stream = get_teacher_response(chain, prompt)
        if response_stream:
            with st.chat_message("assistant"):
                full_response = ""
                message_placeholder = st.empty()
                for chunk in response_stream:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            st.session_state.memory.chat_memory.add_user_message(prompt)
            st.session_state.memory.chat_memory.add_ai_message(full_response)

if __name__ == "__main__":
    main()