import os
import cassio
from dotenv import load_dotenv
from PDF_Downloader import downloader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai

from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

from langchain.prompts import PromptTemplate

#Loading Environment varibales
load_dotenv()

#Configuring the GenAI key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_chunks(input_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000 , chunk_overlap = 1000
    )
    chunks = text_splitter.split_text(input_text)
    return chunks


text = downloader("URL")