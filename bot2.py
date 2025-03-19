import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings  # Keep existing embeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI  # Changed import
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"

# Custom CSS remains the same
st.markdown("""
<style>
    /* Main container styling */
    .stApp {
        background-color: #f0f4f8;
        max-width: 800px;
        margin: auto;
    }
    
    /* Title styling */
    .title-text {
        color: #2b5876;
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* Subtitle styling */
    .subtitle-text {
        color: #4a7ba6;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    /* Chat message styling */
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
    
    /* Input field styling */
    .stChatInput {
        position: fixed;
        bottom: 20px;
        width: 75%;
        left: 50%;
        transform: translateX(-50%);
    }
    
    /* Loading spinner styling */
    .stSpinner > div {
        margin: auto;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_vectorstore():
    # Keep existing embeddings
    embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
    return db

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm():
    # Changed to use DeepSeek API through OpenAI-compatible endpoint
    return ChatOpenAI(
        openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
        model="deepseek-chat",
        base_url="https://api.deepseek.com/v1"
    )

def main():
    # Header section remains the same
    st.markdown('<h1 class="title-text">🩺 Clinical AI Advisor </h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle-text">Your AI-Powered Health Assistant</p>', unsafe_allow_html=True)

    # Initialize chat history remains the same
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I assist you with health-related questions today?"}]

    # Display chat messages remains the same
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="{message["role"]}-message">{message["content"]}</div>', unsafe_allow_html=True)

    # User input remains the same
    prompt = st.chat_input("Ask me anything about health...")

    if prompt:
        # Add user message remains the same
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{prompt}</div>', unsafe_allow_html=True)

        # Generate response
        with st.spinner("Analyzing your question..."):
            try:
                CUSTOM_PROMPT_TEMPLATE = """
                Use the pieces of information provided in the context to answer user's question.
                If you dont know the answer, just say that you dont know, dont try to make up an answer. 
                Dont provide anything out of the given context

                Context: {context}
                Question: {question}
                """

                vectorstore = get_vectorstore()
                qa_chain = RetrievalQA.from_chain_type(
                    llm=load_llm(),
                    chain_type="stuff",
                    retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                    return_source_documents=True,
                    chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
                )

                response = qa_chain.invoke({'query': prompt})
                result = response["result"]

                # Add assistant response remains the same
                st.session_state.messages.append({"role": "assistant", "content": result})
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="assistant-message">{result}</div>', unsafe_allow_html=True)

            except Exception as e:
                error_msg = f" Sorry, I encountered an error. Please try again. ({str(e)})"
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                with st.chat_message("assistant"):
                    st.markdown(f'<div class="assistant-message" style="color: #d32f2f;">{error_msg}</div>', unsafe_allow_html=True)

        # Auto-scroll remains the same
        st.markdown("""
        <script>
            window.addEventListener('DOMContentLoaded', function() {
                const chatContainer = window.parent.document.querySelector('.stApp');
                chatContainer.scrollTop = chatContainer.scrollHeight;
            });
        </script>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()