# RAG for Educational Content 📚
## Overview 📜
RAG for Educational Content is a Retrieval-Augmented Generation (RAG) system built to enhance how students, educators, and researchers access learning materials.
It combines vector search, knowledge graph retrieval, and large language models (LLMs) to provide accurate, context-rich answers from educational documents.

By leveraging both semantic similarity and structured relationships, this project delivers more relevant and well-contextualized information than standard search systems — making it ideal for academic environments.

## Features ✨
- **Document Extraction & Preprocessing** — Reads and cleans educational material from multiple formats.
- **Vector Store Creation (FAISS)** — Generates embeddings and stores them for lightning-fast semantic search.
- **Knowledge Graph Integration** — Builds a graph of concepts and relationships for smarter retrieval.
- **Fusion Retriever** — Combines vector and graph retrieval for maximum accuracy.
- **Intelligent Content Generation** — Produces coherent and fact-rich answers using LLMs.
- **Automated Evaluation** — Uses BERTScore for semantic similarity assessment.

## System Architecture 🏗️

The system follows a *RAG pipeline* enhanced with a *Fusion Retriever* and *Knowledge Graph (KG)*.  
It has *three main stages*:

### 1. Data Ingestion & Processing 📥
- *Document Extraction* (`extractDocs.py`)  
  - Reads educational content (PDFs, text, etc.)  
  - Cleans, tokenizes, and splits into chunks.  

- *Embedding Generation* (`createVectorstore.py`)  
  - Creates embeddings using Transformer models.  
  - Stores them in FAISS.  

- *Knowledge Graph Creation* (`kg.py`)  
  - Extracts entities & relationships, storing them in a graph (NetworkX).  

---

### 2. Retrieval Layer 🔍
- *Vector Retriever* — Finds semantically similar chunks via FAISS.  
- *KG Retriever* (`kg_retrieval.py`) — Retrieves conceptually connected topics via graph traversal.  
- *Fusion Retriever* — Combines both retrieval methods for best results.  

---

### 3. Generation & Evaluation 🧠
- *Answer Generation* (`generation.py`) — LLM creates structured, detailed answers.  
- *Evaluation* (`bertscore.py`) — BERTScore checks similarity with reference content.

  ## Tech Stack 🛠️

**Programming Language**
- Python 3.9+

**Frameworks & Libraries**
- **LangChain** — Framework for building LLM-powered applications  
- **FAISS** — Vector database for fast similarity search  
- **Transformers (Hugging Face)** — For embedding generation  
- **Neo4j** — For Knowledge Graph creation and traversal   
- **BERTScore** — For evaluation of generated content

**Large Language Models**
- Gemma3 via Ollama
- Hugging Face embedding models

**Data Storage**
- FAISS (for vector embeddings)

## Future Scope 🚀

- **Multilingual Support**  
  Enable the system to process and generate content in multiple languages for wider accessibility.

- **Real-time Data Ingestion**  
  Integrate APIs to pull live educational data from trusted sources.

- **Enhanced Knowledge Graph**  
  Incorporate more complex relationships and reasoning capabilities in the KG.
