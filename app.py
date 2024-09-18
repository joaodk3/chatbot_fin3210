import streamlit as st
from openai import OpenAI
import pdfplumber
import re
from openai import OpenAIError

units = {
    "Unit 1 - Introduction": r"assets\Unit1_Intro.pdf",
    'Unit 2 - Financial Statements': r"assets\Unit2_FinancialStatements.pdf", 
    'Unit 3 - Time Value of Money': r"assets\Unit3_TimeValueOfMoney_annotated.pdf", 
    'Unit 4 - Bonds': r"assets\Unit4_BONDS_Annotated.pdf", 
    'Unit 5 - Stocks': r"assets\Unit5_Stocks_SlideDeck_annotated.pdf", 
    'Unit 6 - Capital Budgeting': r"assets/Capital_Budgeting.pdf"
}

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
        model="gpt-4",
        messages=full_messages,
        temperature=0.7,
        max_tokens=500,
        stream=True
    )
    
    return response

def main():
    st.title("🤓🧮 FIN3210 Chatbot")

    with st.sidebar:
        st.image("https://asset4.applyesl.com/images/sch/0134800/04.jpg")
        api_key = st.text_input("🔑 Insert your OpenAI API key", type="password")
        option = st.selectbox("📓 Select the Class Topic", list(units.keys()))
        st.write("You selected:", option)
        st.write("\n")
        st.link_button("Get an OpenAI API Key", "https://platform.openai.com/account/api-keys", type='primary')
            
        if st.button("Clear Conversation"):
            st.session_state.messages = []

    if not api_key:
        st.error("🔒 Please enter your OpenAI API key to continue.")
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