import streamlit as st
from openai import OpenAI
import pdfplumber
import re
from openai import OpenAIError
import os

units = {
    "Unit 1 - Introduction": os.path.join("assets", "Unit1_Intro.pdf"),
    'Unit 2 - Financial Statements': os.path.join("assets", "Unit2_FinancialStatements.pdf"),
    'Unit 3 - Time Value of Money': os.path.join("assets", "Unit3_TimeValueOfMoney_annotated.pdf"),
    'Unit 4 - Bonds': os.path.join("assets", "Unit4_BONDS_Annotated.pdf"),
    'Unit 5 - Stocks': os.path.join("assets", "Unit5_Stocks_SlideDeck_annotated.pdf"),
    'Unit 6 - Capital Budgeting': os.path.join("assets", "Capital_Budgeting.pdf")
}

gpt_models = ["gpt-3.5-turbo", "gpt-4o", "gpt-4"]

@st.cache_data(show_spinner=False)
def read_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    
    # Clean up the text
    text = re.sub(r'\n+', '\n', text)  
    text = re.sub(r'\s+', ' ', text)   
    
    return text

def validate_api_key(api_key):
    client = OpenAI(api_key=api_key)
    try:
        #API call to check if the key is valid
        client.models.list()
        return True, None
    except OpenAIError as e:
        return False, str(e)

def get_teacher_response(client, messages, pdf_content):
    system_message = f"""You are an AI teaching assistant for a finance course. Your role is to answer questions based solely on the content provided from the selected PDF. Here's a summary of the PDF content:

{pdf_content[:100000]}... (truncated for brevity)

Please adhere to the following guidelines:
1. Only answer questions related to the content in the provided PDF summary.
2. If a question is outside the scope of the PDF content, politely inform the student that the topic is not covered in the current unit and suggest they refer to the appropriate unit or ask their professor.
3. Use a friendly, professional tone appropriate for a teaching assistant.
4. If you're unsure about an answer, it's okay to say so rather than providing potentially incorrect information.
5. When discussing formulas or equations, present them clearly and explain their components.
"""

    full_messages = [
        {"role": "system", "content": system_message},
        *messages
    ]

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=0.7,
        max_tokens=500,
        stream=True
    )
    
    return response

def faq_section():
    st.markdown("### Frequently Asked Questions (FAQ)")

    with st.expander("How Does the App Work?"):
        st.write("""
        This app allows you to ask questions about Managerial Finance topics. The chatbot is powered by OpenAI's GPT models, 
        which read the selected unit's PDF content of the class and provide answers based on it. Make sure to provide a valid API key 
        to interact with the chatbot.
        """)

    with st.expander("How to get an OpenAPI Key?"):
        st.write("""
        You can get an OpenAI API key by signing up at [OpenAI's API platform](https://platform.openai.com/signup). 
        Once you've signed up, navigate to the API section in your account to generate an API key.
        """)

    with st.expander("What is an API Key?"):
        st.write("""
        An API key is a unique identifier used to authenticate requests made to the OpenAI API. It allows your app 
        to securely interact with OpenAI's models.
        """)

    with st.expander("Is the API Key Free?"):
        st.write("""
        OpenAI offers limited free access to its API, but usage beyond certain limits will require a paid subscription.
        You can check their pricing on the [OpenAI pricing page](https://openai.com/pricing).
        """)

    with st.expander("What are the differences between GPT models?"):
        st.write("""
        - **GPT-3.5-turbo**: The most cheap model, cost-effective, designed for more common questions.
        - **GPT-4o**: A more advanced variant of GPT-4, which is faster and optimized for specific tasks.
        - **GPT-4**: The most advanced model offering higher accuracy and a better understanding of complex inputs, 
        but is more expensive and may take more time to generate responses, recommended for questions involving 
        complex mathematical questions.
        """)
    
    with st.expander("How to keep track of your API credits usage?"):
            st.write('''
            You can track your OpenAI API usage by visiting the [Usage Dashboard](https://platform.openai.com/account/usage). 
            It provides details on your current usage and remaining credits.
            ''')

def main():
    st.title("🤓🧮 FIN3210 Chatbot")

    with st.sidebar:
        st.image("https://asset4.applyesl.com/images/sch/0134800/04.jpg")
        api_key = st.text_input("🔑 Insert your OpenAI API key", type="password")
        global model
        model = st.selectbox("Select a GPT Model", gpt_models)
        option = st.selectbox("📓 Select the Class Topic", list(units.keys()))
        st.write("You selected:", option)
        st.divider()
        faq_section()
        st.divider()
        if st.button("Clear Conversation", type="secondary"):
            st.session_state.messages = []

    if not api_key:
        st.error("🔒 Please enter your OpenAI API key to continue.")
        st.link_button("Get an OpenAI API Key", "https://platform.openai.com/account/api-keys", type='secondary')
        return
    
    # validate the API key
    is_valid, error_message = validate_api_key(api_key)
    if not is_valid:
        st.error(f"❌ Invalid API key, please make sure you get a valid API Key: https://platform.openai.com/account/api-keys")
        return

    client = OpenAI(api_key=api_key)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question about the selected unit"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        pdf_content = read_pdf(units[option])

        with st.chat_message("assistant"):
            stream = get_teacher_response(client, st.session_state.messages, pdf_content)
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()