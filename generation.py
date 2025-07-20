import pandas as pd
import asyncio
from evaluate import load
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough



llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
prompt_hub_rag = hub.pull("rlm/rag-prompt")
import torch 

print ("cuda avaialable:" ,torch.cuda.is_available())

async def generate_answer(question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template ="""Answer the question based only on the following context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    print("Running the chain")
    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
    return rag_chain.invoke(question)

async def main():
    df = pd.read_excel("Copy of RAG_task2_8thJune_answers(1).xlsx")
    df['generated'] = await asyncio.gather(*(generate_answer(q) for q in df['Queries']))

    print(df['generated'])

asyncio.run(main())
