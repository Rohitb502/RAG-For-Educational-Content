from langchain.docstore.document import Document
from typing import List
from rank_bm25 import BM25Okapi
import numpy as np
from extractDocs import fetch_pdf_chunks
from langchain.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

cleaned_texts = fetch_pdf_chunks()
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)

def bm25_index(documents: List[Document]):
    tokenized_docs = [doc.page_content.split() for doc in documents]
    return BM25Okapi(tokenized_docs)

bm25 = bm25_index(cleaned_texts)

def fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5):
    epsilon = 1e-8

    bm25_scores = bm25.get_scores(query.split())  
    
    # Get all docs and their vector scores
    all_docs = vectorstore.similarity_search("", k=vectorstore.index.ntotal)
    vector_results = vectorstore.similarity_search_with_score(query, k=len(all_docs))
    
    # Map document content to index
    doc_to_index = {doc.page_content: idx for idx, doc in enumerate(all_docs)}
    
    vector_indices = []
    vector_scores = []
    for doc, score in vector_results:
        if doc.page_content in doc_to_index:
            idx = doc_to_index[doc.page_content]
            vector_indices.append(idx)
            vector_scores.append(score)

    vector_scores = np.array(vector_scores)
    vector_scores = 1 - (vector_scores - np.min(vector_scores)) / (np.max(vector_scores) - np.min(vector_scores) + epsilon)

    bm25_subset_scores = np.array([bm25_scores[i] for i in vector_indices])
    bm25_subset_scores = (bm25_subset_scores - np.min(bm25_subset_scores)) / (np.max(bm25_subset_scores) -  np.min(bm25_subset_scores) + epsilon)

    combined_scores = alpha * vector_scores + (1 - alpha) * bm25_subset_scores
    sorted_indices = np.argsort(combined_scores)[::-1]

    return [all_docs[vector_indices[i]] for i in sorted_indices[:k]]


query = "What are the different types of a system: a second order and first order system. Give example"

# Perform fusion retrieval
top_docs = fusion_retrieval(vectorstore, bm25, query, k=5, alpha=0.5)
docs_content = [doc.page_content for doc in top_docs]
print(docs_content)