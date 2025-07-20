import pandas as pd
import ast
import asyncio
from ragas import SingleTurnSample
from ragas.metrics import NonLLMContextPrecisionWithReference
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index_folder", embeddings=embedding_model)

def retrieve_context(question, vectorstore):
    
    retrivedDocs = []
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(question)
    retrivedDocs.extend(docs)
    return retrivedDocs

# Load Excel file
df = pd.read_excel("RAG_TEST.xlsx")  #

df['reference'] = df['reference'].apply(lambda x: [x])

# Run retrieval for each question
df['retrieved'] = df['question'].apply(retrieve_context)

context_precision = NonLLMContextPrecisionWithReference()

# Async function to score all samples
async def compute_precision(dataframe):
    scores = []
    for _, row in dataframe.iterrows():
        sample = SingleTurnSample(
            retrieved_contexts=row['retrieved'],
            reference_contexts=row['reference']
        )
        score = await context_precision.single_turn_ascore(sample)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

# Run the async scoring
average_score = asyncio.run(compute_precision(df))

print(" Overall Context Precision Score:", average_score)
