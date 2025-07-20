import pandas as pd
import asyncio
from evaluate import load
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain import hub
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from ragas.dataset_schema import SingleTurnSample 
from ragas.metrics import ResponseRelevancy

llm = Ollama(model="deepseek-r1:1.5b", temperature=0)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.load_local("faiss_index_folder", embeddings=embedding_model)
prompt_hub_rag = hub.pull("rlm/rag-prompt")

def generate_answer(question):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    template = """Answer the question based only on the following context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
    return rag_chain.invoke(question)


df = pd.read_excel("RAG_TEST.xlsx")
df['reference'] = df['reference'].apply(lambda x: [x])
df['retrieved'] = df['question'].apply(generate_answer)

async def compute_faithfulness(dataframe):
    scores = []
    for _, row in dataframe.iterrows():
        sample = SingleTurnSample(
            retrieved_contexts=row['retrieved'],
            reference_contexts=row['reference']
        )
        score = await ResponseRelevancy(llm=llm, embeddings=embedding_model)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

# Run the async scoring
average_score = asyncio.run(compute_faithfulness(df))

print(" Overall Context Precision Score:", average_score)