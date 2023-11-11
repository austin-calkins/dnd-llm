import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from flask import Flask
from flask import request

app = Flask(__name__)

load_dotenv()

prompt_template = """

Use the following context to answer the question. 

Say "Sorry I don't know the answer" if you don't know the answer. 

{context}

Question: {question}
"""

prompt = PromptTemplate.from_template(prompt_template)

def get_chunks(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, chunk_overlap=50)
    
    chunks = splitter.split_documents(data)

    return chunks

def get_db_retriever(chunks):
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(chunks,embeddings)

    return db.as_retriever()

def load_data():
    print("Loading data from PDF document")
    pdf_loader = PyPDFLoader("./docs/DnD-5e-Players-Handbook.pdf")
    data = pdf_loader.load()

    return data

def retrieval_qa(q):
    
    largeLanguageModel = OpenAI()

    retrievalQA = RetrievalQA.from_chain_type(largeLanguageModel, 
        chain_type="stuff", 
        retriever=app.global_db_retriever,
        chain_type_kwargs={"prompt": prompt})

    return retrievalQA({"query": q})

def ask(question):
    return retrieval_qa(question)

@app.route("/call",methods=["POST"])
def call():
    requestJson = request.json
    print("1. Input is: ",requestJson)
    question = requestJson.get('question')

    response = ask(question)
   
    return response


# Setting data in the application context outside of a request
with app.app_context():
    print("generating seed data from players handbook")
    data = load_data()
    chunks = get_chunks(data)
    app.global_db_retriever = get_db_retriever(chunks)


if __name__== '__main__':
    app.run(host="0.0.0.0",port=3099)