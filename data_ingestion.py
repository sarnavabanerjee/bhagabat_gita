import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# 1. Load all PDFs from the Data directory
loader = DirectoryLoader(path='Data', glob='*.pdf', show_progress=True)
documents = loader.load()

# 2. Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("vectorstore")
print("Vectorstore saved to ./vectorstore")