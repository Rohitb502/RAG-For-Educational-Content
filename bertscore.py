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

async def main():
    df = pd.read_excel("RAG_TEST.xlsx")
    df['reference'] = df['reference'].apply(lambda x: [x])
    df['generated'] = await asyncio.gather(*(generate_answer(q) for q in df['Queries']))

    avg_score = await compute_bertscore(df)
    print(avg_score)


async def compute_bertscore(dataframe):
    bertscore = load("bertscore")
    for _, row in dataframe.iterrows():
        predictions=row['retrieved']
        references=row['reference']
        result = bertscore.compute(predictions=predictions, references=references, lang="en")

    precision = sum(result['precision']) / len(result['precision']) if result['precision'] else 0.0
    recall = sum(result['recall']) / len(result['recall']) if result['recall'] else 0.0
    f1 = sum(result['f1']) / len(result['f1']) if result['f1'] else 0.0

    return {
        'precision' : precision,
        'recall' : recall,
        'f1' : f1
    }

if __name__ == "__main__":
    asyncio.run(main())


