from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

from extractDocs import fetch_pdf_chunks

try:
    # Initialize embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    
    # Fetch documents
    print("Fetching PDF chunks from database...")
    documents_to_process = fetch_pdf_chunks()
    
    if documents_to_process:
        print(f"Successfully fetched {len(documents_to_process)} documents.")
        print("Creating FAISS vectorstore...")
        
        # Create a vectorstore from documents
        vectorstore = FAISS.from_documents(documents_to_process, embedding_model)
        
        # Save the vectorstore
        vectorstore.save_local("faiss_index")
        print("FAISS vectorstore created and saved successfully!")
        
    else:
        print("No documents were fetched. Please check your database connection and data.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()