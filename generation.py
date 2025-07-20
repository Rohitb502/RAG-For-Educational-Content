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
from retriever import fusion_retrieval
from retriever import bm25_index
from extractDocs import fetch_pdf_chunks

llm = OllamaLLM(model="deepseek-r1:1.5b", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding_model, allow_dangerous_deserialization=True)
prompt_hub_rag = hub.pull("rlm/rag-prompt")
cleaned_texts = fetch_pdf_chunks()
bm25 = bm25_index(cleaned_texts)

async def generate_answer(question):
    retriever = fusion_retrieval(vectorstore, bm25, question, k=5, alpha=0.5)
    template = """You are an expert in Control System Design and can explain content very well
    using presentations when given content about the topic to cover in the Presentation.
    Create a presentation based on the following query and the content provided.You should
    strictly follow the content which is given and not try to use any other information.
    Create a Presentation by following the steps given below:

    1) Analyze the given question and think about what topics need to be covered in the presentations and they should reach the 
    learning goals provided in the query.   
    2) Match the topics which needs to be covered to the following content provided.
    3) Extract all the information which relevant to make a presnentations on the given topics such that topics will be well covered
    and all the information will be available in the presenatation
    4) The first slide should provide a small overview of the topics which will be covered in the presentation
    5) In the following slides which you will create 2-3 slides for each topic which needs to be covered. In each slide there should
    be only 5-6 bullet points.
    6) Follow the given format for each slide
    Slide Title: [Title Here]

    - Bullet Point 1
    - Bullet Point 2
    - Bullet Point 3

    7) Make sure in the each slide of the presentation only major points are covered and do not go into the details of the topic. The
    points should should be short and impactful.Avoid lengthy paragraphs and focus on clarlity of each point.
    8) Add a conclusion slide in the end which will should conclude the presentation and how each topic covered will be related to 
    each other and focus on how they meet the learning goal. 

    Query: {question}
    Content: {context}"""
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
