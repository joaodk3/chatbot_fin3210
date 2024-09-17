import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader

# May become a dict with name and file path
units = {"Unit 1 - Introduction": r"assets\Unit1_Intro.pdf",
         'Unit 2 - Financial Statements': r"assets\Unit2_FinancialStatements.pdf", 
         'Unit 3 - Time Value of Money': r"assets\Unit3_TimeValueOfMoney_annotated.pdf", 
         'Unit 4 - Bonds': r"assets\Unit4_BONDS_Annotated.pdf", 
         'Unit 5 - Stocks': r"assets\Unit5_Stocks_SlideDeck_annotated.pdf", 
         'Unit 6 - Capital Budgeting': r"assets/Capital_Budgeting.pdf"
        }

def read_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

def main():

    # Sidebar for API key and unit selection
    with st.sidebar:
        api_key = st.text_input("Insert your free OpenAI API key (https://platform.openai.com/api-keys)", type="password")
        option = st.selectbox("Select the Class Topic", list(units.keys()))
        st.write("You selected:", option)

        if st.button("Clear"):
            st.session_state.messages = []
            st.session_state.selected_option = None

    # Error checking for API key
    if not api_key:
        st.error("Please enter your OpenAI API key to continue.")
        return
    
    st.title("Welcome to FIN3210 Chatbot")

    client = OpenAI(api_key=api_key)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4o"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("What is up?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        pdf_content = read_pdf(units[option])

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()


#Still answering questions from outside of the pdf content - role is not defined
