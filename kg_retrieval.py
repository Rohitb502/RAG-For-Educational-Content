from extractDocs import fetch_pdf_chunks
from langchain_community.graphs import Neo4jGraph
import os
from langchain_core import documents
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_ollama.llms import OllamaLLM
import pandas as pd
import asyncio

NEO4J_URI="neo4j+s://5cf0adc7.databases.neo4j.io"
NEO4J_USERNAME="neo4j"
NEO4J_PASSWORD="h-OZAMdhUCJpYI5jO-FJYf9lbmjNM4y8vQNve3RKKgw"
os.environ["NEO4J_URI"]=NEO4J_URI
os.environ["NEO4J_USERNAME"]=NEO4J_USERNAME
os.environ["NEO4J_PASSWORD"]=NEO4J_PASSWORD

graph=Neo4jGraph(
    url=NEO4J_URI,
    username=NEO4J_USERNAME,
    password=NEO4J_PASSWORD,
)

llm = OllamaLLM(model = "deepseek-r1:1.5b", temperature=0)
async def generate_answer(question):
    chain = GraphCypherQAChain.from_llm(
        graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True
    )
    return chain.invoke({"query": question})

async def main():
    df = pd.read_excel("Copy of RAG_task2_8thJune_answers(1).xlsx")
    df['generated'] = await asyncio.gather(*(generate_answer(q) for q in df['Queries']))
    print(df['generated'])