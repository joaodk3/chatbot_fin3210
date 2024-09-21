# FIN3210 Chatbot

## Overview

This is a chatbot application built using **Streamlit** and **OpenAI**'s GPT models. The chatbot is designed to assist students in a Managerial Finance course by answering questions based on provided PDF content from the course units. The application reads the content of the selected PDF, splits the text into manageable chunks, and uses OpenAI's GPT models to generate responses in a conversational format.

## Features

- **PDF Integration**: The application reads and processes PDFs related to specific finance topics.
- **Chunk Text**: Uses a recursive text splitter to handle large documents efficiently.
- **Chatbot Functionality**: Powered by GPT models (gpt-3.5-turbo, gpt-4o, gpt-4), which respond to questions based on the content of the selected unit.
- **API Rate Limiting**: A custom rate limiter ensures compliance with OpenAI's rate limits.
- **Conversation Memory**: Uses Langchain's `ConversationBufferMemory` to maintain context throughout the conversation.
- **User-friendly Interface**: A simple Streamlit-based UI with sidebar options for API key input, topic selection, and model selection.

## Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Set up assets: Ensure that the necessary PDF files are placed in the `assets` directory. The application expects the following files:

   ```
   assets/
   ├── Unit1_Intro.pdf
   ├── Unit2_FinancialStatements.pdf
   ├── Unit3_TimeValueOfMoney_annotated.pdf
   ├── Unit4_BONDS_Annotated.pdf
   ├── Unit5_Stocks_SlideDeck_annotated.pdf
   └── Capital_Budgeting.pdf
   ```

5. Run the application:

   ```bash
   streamlit run app.py
   ```

## Usage

1. **API Key**: You need to input your OpenAI API key to use the application. If you do not have an API key, you can sign up at [OpenAI's API platform](https://platform.openai.com/signup).
   
2. **Selecting a Topic**: From the sidebar, select a topic from the available course units.
   
3. **Asking Questions**: Type in your question related to the selected unit, and the chatbot will respond based on the content of the corresponding PDF file.
   
4. **Clearing Conversation**: You can clear the conversation history at any time by clicking the "Clear Conversation" button.

## FAQ

### How Does the App Work?

The chatbot uses OpenAI's GPT models to answer questions about Managerial Finance topics. It reads the selected unit's PDF content and generates responses based on it. You can interact with the chatbot by providing a valid API key.

### How to Get an OpenAI Key?

Sign up at [OpenAI's API platform](https://platform.openai.com/signup), then navigate to the API section to generate an API key.

### What is an API Key?

An API key is a unique identifier used to authenticate requests made to OpenAI’s API.

### Is the API Key Free?

OpenAI offers limited free access to its API, but usage beyond a certain limit requires a paid subscription.

### How to Track API Usage?

Track your usage on the [Usage Dashboard](https://platform.openai.com/account/usage).

## License

This project is licensed under the MIT License.