import os
import torch
import transformers
from transformers import pipeline
from PDF_Downloader import downloader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

model_token = os.environ.get("HUGGINGFACEAPI_TOKEN")

# Function to creat Text Chunks 
def get_text_chunks(input_text):
    document_splitter = CharacterTextSplitter(chunk_size = 500 , chunk_overlap=0)
    document_chunks = document_splitter.split_documents(document)
    return document_chunks

def get_vector_store(document_chunks):
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(document_chunks , embedding = embeddings , persist_directory = "./data")
    
    return vectordb


def get_conversational_chain():

    model = ChatGoogleGenerativeAI(model="gemini-pro" , temperature=0.3) 
    promt = PromptTemplate(template=prompt_template , input_variables=["context" , "question"])
    chain = load_qa_chain(model , chain_type="stuff" , promt=promt)

    return chain

