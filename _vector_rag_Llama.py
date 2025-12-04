import os
import pandas as pd
import json
import torch

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import HuggingFacePipeline
from datasets import load_from_disk
# from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever
# from langchain.retrievers.document_compressors import CrossEncoderReranker

from utils import load_datasets, dense_retriever, cross_encoder

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from dotenv import load_dotenv
from huggingface_hub import login
from utils import load_llama_model, load_embedding_model

# GPU 3번
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
print("sys.executable:", __import__("sys").executable)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU count (in this notebook):", torch.cuda.device_count())
    print("This notebook sees GPU:", torch.cuda.get_device_name(0))

# 경로 & 하이퍼파라미터 설정
CORPUS_PATH = "/mnt/aix23604/hotpotqa/corpus_distractor.parquet" # 코퍼스 파일 위치
FAISS_PATH = "/mnt/aix23604/rag/llama_faiss_hotpotqa_index" # FAISS 인덱스 위치
EMB_MODEL = "BAAI/bge-base-en-v1.5"  # 임베딩 모델 이름 : "BAAI/bge-large-en-v1.5", "all-MiniLM-L6-v2" [tuple에서 문자열로 수정]
retriever_k = 5 # retriever에서 가져올 문서 개수
cross_encoder_k = 3 # cross-encoder로 다시 rerank할 문서 개수
OUTPUT_PATH = "/mnt/aix23604/output"
RESULT_PATH = os.path.join(OUTPUT_PATH, 'llama_validation_sample_10.json')

# Llama Model
load_dotenv(override=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
login(token=HF_TOKEN)

llama_model_id = "meta-llama/Llama-3.1-8B-Instruct"
llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

llama_model = AutoModelForCausalLM.from_pretrained(
    llama_model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
)
llama_model.to(device)

# Llama를 LangChain pipeline으로 매핑
text_generation_pipeline = pipeline(
    "text-generation",
    model=llama_model,
    tokenizer=llama_tokenizer,
    max_new_tokens=256,
    do_sample=True,
    temperature=0.3,
    top_p=0.9,
    repetition_penalty=1.2,
    device=0 if device == "cuda" else -1
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# llama_model = load_llama_model()

# RAG chain 구성
template = """
Answer the question based on the following pieces of context:

Context:
{context}

Question: {question}

Answer:
"""

prompt = ChatPromptTemplate.from_template(template)

# 코퍼스 로드 + 임베딩 + FAISS 로드
# df_corpus = pd.read_parquet(CORPUS_PATH, EMB_MODEL)
df_corpus = pd.read_parquet(CORPUS_PATH) # corpus_distractor.parquet 파일 로드
embeddings = load_embedding_model()
vectorstore = FAISS.load_local(FAISS_PATH, embeddings, allow_dangerous_deserialization=True) # FAISS 벡터스토어 로드
retriever = cross_encoder(retriever_k, cross_encoder_k, df_corpus, vectorstore, emb_model=EMB_MODEL)

# embeddings = HuggingFaceEmbeddings(
#     model_name=EMB_MODEL, # bge-base 임베딩 모델 로드 (GPU 사용, 벡터 정규화)
#     model_kwargs={'device': 'cuda'}, 
#     encode_kwargs={
#         'normalize_embeddings': True }
# )



# RAG 체인 정의
def format_docs(docs):   
    return "\n\n".join([d.page_content for d in docs])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# HotpotQA Distractor 데이터셋으로 평가용 결과 생성
# Result
ds_dataset = load_datasets('distractor')
ds_dataset = ds_dataset['validation']
ds_dataset = ds_dataset.select(range(10))

# ds_path = "/mnt/aix23604/hotpotqa/distractor"
# ds_dataset = load_from_disk(ds_path)

# print("Dataset type:", type(ds_dataset))
# print("Dataset keys:", ds_dataset.keys() if hasattr(ds_dataset, 'keys') else "No keys")
# print("First item:", ds_dataset[0] if len(ds_dataset) > 0 else "Empty dataset")


print(f"Processing {len(ds_dataset)} samples...")
def clean_answer(answer):
    # "Answer:" 이후만 추출
    if "Answer:" in answer:
        parts = answer.split("Answer:")
        answer = parts[-1].strip()
    
    # "Human:" 이전만 추출
    if "Human:" in answer:
        answer = answer.split("Human:")[0].strip()
    
    # "Question:" 이전만 추출
    if "Question:" in answer:
        answer = answer.split("Question:")[0].strip()
    
    # 반복 제거
    sentences = answer.split('.')
    cleaned_sentences = []
    last_sentence = ""
    repeat_count = 0
    
    for sent in sentences:
        sent = sent.strip()
        if sent == last_sentence:
            repeat_count += 1
            if repeat_count >= 2:  # 2번 반복되면 중단
                break
        else:
            repeat_count = 0
            last_sentence = sent
            if sent:
                cleaned_sentences.append(sent)
    
    answer = '. '.join(cleaned_sentences)
    if answer and not answer.endswith('.'):
        answer += '.'
    
    return answer.strip()

results = []

for idx, row in enumerate(ds_dataset):
    # row = ds_dataset[idx]
    question = row['question']
    ground_truth = row['answer']

    # 검색된 문서 내용도 저장해야 RAGAS 평가 가능
    retrieved_docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in retrieved_docs]
    retrieved_titles = [doc.metadata.get('title', '') for doc in retrieved_docs]

    # 답변 생성
    answer = rag_chain.invoke(question)
    
    answer = clean_answer(answer)

    # if "Answer:" in answer:
    #     answer = answer.split("Answer:")[-1].strip()
    # if "Human:" in answer:
    #     answer = answer.split("Human:")[0].strip()
    # answer = answer.strip()

    results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,     
            "retrieved_titles": retrieved_titles,
            "ground_truth": ground_truth
        })
    if (idx + 1) % 10 == 0:
        print(f"Processed {idx + 1} questions")

result_df = pd.DataFrame(results)
os.makedirs(OUTPUT_PATH, exist_ok=True)
result_df.to_json(RESULT_PATH, orient='records', force_ascii=False, indent=2)
print(f"Results saved to {RESULT_PATH}")

print("\nSample answer:")
print(result_df['answer'].iloc[0])