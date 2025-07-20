import mysql.connector
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

  
files = ["books/Mordern Control Systems.pdf", 
         "books/Finite Dimensional Linear Systems.pdf",
         "books/Robust process control.pdf",
         "books/Linear Controller Design Limits of Performance.pdf",
         "books/Control-System-Design.pdf"]


all_docs = []

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300,
    chunk_overlap=50)

for file in files:
    loader = PyPDFLoader(file)
    print(f"Loading {file}")
    docs = loader.load()
    all_docs.extend(docs)

splits = text_splitter.split_documents(all_docs)

# Connect to MySQL
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Password123@",
    database="content"
)
cursor = conn.cursor()

query = """
INSERT INTO pdf_chunks (content, file_name, page_number)
VALUES (%s, %s, %s)
"""

for doc in all_docs:
    
    content = doc.page_content
    file_name = doc.metadata.get("source", "unknown")
    page_number = doc.metadata.get("page", 0)

    cursor.execute(query, (content, file_name, page_number))

conn.commit()
cursor.close()
conn.close()



