import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS for styling
st.markdown("""
<style>
    .stApp {
        background-color: #f0f4f8;
        max-width: 800px;
        margin: auto;
    }
    .title-text {
        color: #2b5876;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle-text {
        color: #4a7ba6;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 12px;
        margin: 5px 0;
        max-width: 80%;
        float: right;
        clear: both;
    }
    .assistant-message {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 12px;
        margin: 5px 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 75%;
        left: 50%;
        transform: translateX(-50%);
    }
    .stSpinner > div {
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(model_name):
    return OllamaLLM(model=model_name)

def clean_response(response):
    return response.strip()

def main():
    # Header section
    st.markdown('<h1 class="title-text">🩺 Clinical AI Advisor </h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Your AI-Powered Health Assistant</p>', unsafe_allow_html=True)

    # Initialize chat history if not already present
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hello! How can I assist you today with your health-related questions?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="{message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)

    # User input
    prompt = st.chat_input("Feel free to ask me anything about health...")

    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

        # Generate response
        with st.spinner("Processing..."):
            try:
                CUSTOM_PROMPT_TEMPLATE = """
                You are an expert health assistant. 
                Answer the user's question using only the information from the provided context. 
                Do not generate additional questions or content beyond the dataset. only maintain formalities.
                Ensure responses are friendly, empathetic, and professional. 

                **Context**: {context}

                **Question**: {question}

                **Answer**:
                """

                model_name = "llama2"  # Replace with your specific Llama model name
                vectorstore = get_vectorstore()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(model_name),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3, 'filter': {'dataset': 'your_dataset_name'}}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]

                # Clean the response
                result = clean_response(result)

                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": result})
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="assistant-message">{result}</div>', unsafe_allow_html=True)

            except Exception as e:
                error_msg = "Sorry, I couldn't process your request. Please try again later."
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="assistant-message">{error_msg}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
