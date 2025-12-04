'''
corpus 호출 -> chunk -> faiss 
EMB_MODEL 상의해서 결정 & 실행
'''

CORPUS_PATH = "/mnt/aix23604/hotpotqa/corpus_distractor.parquet"
FAISS_PATH = "/mnt/aix23604/hotpotqa/faiss_distractor" 
EMB_MODEL = "BAAI/bge-base-en-v1.5",  # "BAAI/bge-large-en-v1.5", "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
OVERLAP = 30
# GPU 설정 필요 

import pandas as pd
from tqdm import tqdm
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_corpus (load_path):
    df = pd.read_parquet(load_path)
    print(f"로드 완료: {len(df)}개의 문서")
    return df

def generat_document(load_path):
    df = load_corpus (load_path)
    documents = []
    for _, row in df.iterrows():
        doc = Document(  # from langchain_core.documents import Document
            page_content=row['text'],
            metadata={"title": row['title'], "source_id": row.get("source_id")}
        )
        documents.append(doc)
    return documents

def generate_faiss(documents, save_path, chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP):
    text_splitter = RecursiveCharacterTextSplitter( # from langchain_text_splitters import RecursiveCharacterTextSplitter
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
    splits = text_splitter.split_documents(documents)
    total_chunks= len(splits)
    print(f"   -> 총 {total_chunks}개의 청크(Chunk) 생성됨")

    # embeddings = OpenAIEmbeddings()  # L2 distance # from langchain_openai import OpenAIEmbeddings
    embeddings = HuggingFaceEmbeddings(
    model_name=  EMB_MODEL
    model_kwargs={'device': 'cuda'}, 
    encode_kwargs={'normalize_embeddings': True, 'batch_size': 2048}
    )
    
    text_list = [d.page_content for d in splits]
    metadatas = [d.metadata for d in splits]
    process_batch_size = 50000
    
    start_time = time.time()

    vectorstore = FAISS.from_texts(
        texts=text_list[:100],
        embedding=embeddings,
        metadatas=metadatas[:100]
    )

    for i in tqdm(range(100, total_chunks, process_batch_size), desc="Indexing Progress"):
        end_i = min(i + process_batch_size, total_chunks)
        
        # 슬라이싱
        batch_texts = text_list[i:end_i]
        batch_metas = metadatas[i:end_i]
    
        vectorstore.add_texts(texts=batch_texts, metadatas=batch_metas)
        
    elapsed_time = (time.time() - start_time) / 60
    print(f"   -> 완료! 소요 시간: {elapsed_time:.1f}분")

    vectorstore.save_local(save_path)
    print("   -> 저장 완료.")
    
    return vectorstore

documents = generat_document(CORPUS_PATH)
generate_faiss(documents, FAISS_PATH)