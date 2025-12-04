'''
rag chain 
gpt-env 추가 필요

'''
import os
import pandas as pd
import json
import torch

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker

from langchain_community.cross_encoders import HuggingFaceCrossEncode


from utils import sparse_retriever, dense_retriever, hybrid_retriever, cross_encoder, load_datasets

os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
print("sys.executable:", __import__("sys").executable)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count (in this notebook):", torch.cuda.device_count())
    print("This notebook sees GPU:", torch.cuda.get_device_name(0))

load_dotenv()
openai_api_key = os.environ.get("OPENAI_API_KEY")

CORPUS_PATH = "/aix23604/hotpotqa/corpus_distractor.parquet"
FAISS_PATH = "/aix23604/hotpotqa/faiss_distractor" 
EMB_MODEL = "BAAI/bge-base-en-v1.5" # "BAAI/bge-large-en-v1.5", "all-MiniLM-L6-v2", "text-embedding-3-small"
retriever_k = 5
cross_encoder_k = 3
OUTPUT_PATH = "/aix23604/output"
RESULT_PATH = os.path.join(OUTPUT_PATH, 'sample.json')

template = """
Answer the question based on the following pieces of context:
{context}

Question: {question}
"""

print("Loading corpus")
df_corpus = pd.read_parquet(CORPUS_PATH, EMB_MODEL)

print("Loading embeddings")
embeddings = HuggingFaceEmbeddings(
    model_name=EMB_MODEL,
    model_kwargs={'device': 'cuda'}, 
    encode_kwargs={
        'normalize_embeddings': True }
)

print("Loading vectorstore")
vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True)

print("Loading LLM")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

print("Setting up retrievers")
# hybrid_retriever = hybrid_retriever(retriever_k)
retriever = cross_encoder(
    retriever_k = retriever_k,
    cross_encoder_k = cross_encoder_k,
    df_corpus = df_corpus,
    vectorstore = vectorstore,
    emb_model = EMB_MODEL
)

# RAG Chain
prompt = ChatPromptTemplate.from_template(template)

# compression_retriever =cross_encoder(EMB_MODEL, cross_encoder_k)

def format_docs(docs):   # LLM에 넣기 좋게 string으로 변경
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": compression_retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


'''
results
'''

print("\nLoading Dataset")
ds_dataset = load_datasets('distractor')
train_small = ds_dataset["train"].shuffle(seed=42).select(range(int(len(ds_dataset["train"])*0.05)))
print(f"Processing {len(train_small)} samples")

# 결과 생성
results = []

for idx, row in enumerate(train_small):
    question = row['question']
    ground_truth = row['answer']

    # 검색된 문서 내용도 저장해야 RAGAS 평가 가능
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    retrieved_titles = [doc.metadata.get('title', '') for doc in retrieved_docs]

    # 답변 생성
    answer = rag_chain.invoke(question)

    results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,     # 리스트 형태여야 함
            "retrieved_titles": retrieved_titles,
            "ground_truth": ground_truth # 문자열
        })

os.makedirs(OUTPUT_PATH, exist_ok=True)
result_df = pd.DataFrame(results)
result_df.to_json(RESULT_PATH, orient='records', force_ascii=False)

print(f"\n Results saved to {RESULT_PATH}")
print(f"Total samples: {len(results)}")
