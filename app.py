import streamlit as st
import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv
load_dotenv()

st.sidebar.title('RAG Gita App')
st.sidebar.markdown('Ask questions about the Bhagavad Gita PDF using Retrieval-Augmented Generation (RAG).')
st.sidebar.info('Powered by LangChain, Gemini, and FAISS')

st.title('📖 RAG Q&A: Bhagavad Gita')
st.markdown('Ask any question about the Bhagavad Gita. The app will retrieve relevant context from the PDF and answer using Google Gemini.')

@st.cache_resource(show_spinner=True)
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={'k': 3})
    template = """
    Use the following pieces of information to answer the user's question. Add your own knowledge if helpful. If you don't know the answer, just say you don't know.
    Context: {context}
    Question: {question}
    Only return the helpful answer below and nothing else.
    """
    PROMPT = ChatPromptTemplate.from_template(template)
    model = ChatGoogleGenerativeAI(model='models/gemini-1.5-flash')
    parser = StrOutputParser()
    chain = (
        RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
        | PROMPT
        | model
        | parser
    )
    return chain, retriever

chain, retriever = load_rag_chain()
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

with st.form(key='chat_form', clear_on_submit=True):
    user_input = st.text_input('Ask a question about the Bhagavad Gita:', '')
    submit = st.form_submit_button('Ask')

if submit and user_input:
    with st.spinner('Thinking...'):
        answer = chain.invoke(user_input)
    st.session_state['messages'].append({'role': 'user', 'content': user_input})
    st.session_state['messages'].append({'role': 'assistant', 'content': answer})

for msg in st.session_state['messages']:
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    else:
        st.markdown(f"**GitaBot:** {msg['content']}")

