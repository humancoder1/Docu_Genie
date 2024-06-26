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

#Inintializing the Connection to the Database
cassio.init(token=os.getenv("ASTRA_DB_APPLICATION_TOKEN") , database_id=os.getenv("ASTRA_DB_ID"))

# Function to creat Text Chunks 
def get_text_chunks(input_text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 10000 , chunk_overlap = 1000
    )
    chunks = text_splitter.split_text(input_text)
    return chunks

def get_vector_store(input_text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    astra_vector_store = Cassandra(
        embedding= embeddings , table_name="vector_table" , session=None , keyspace=None
    ) 
    astra_vector_store.add_texts(input_text_chunks)
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    return astra_vector_index



    return astra_vector_index


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details , if the answer is not in provided
    context just say , "answer is not available in the context , don't provide the wrong answer\n\n 
    Context:\n {context}?\n
    Question:\n {question}\n

    Answer:
"""

    model = ChatGoogleGenerativeAI(model="gemini-pro" , temperature=0.3) 
    promt = PromptTemplate(template=prompt_template , input_variables=["context" , "question"])
    chain = load_qa_chain(model , chain_type="stuff" , promt=promt)

    return chain

