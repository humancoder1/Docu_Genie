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

