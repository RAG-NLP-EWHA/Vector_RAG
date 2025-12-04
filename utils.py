import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from huggingface_hub import login

from datasets import load_from_disk
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# 추가 import 시도
try:
    from langchain.retrievers.ensemble import EnsembleRetriever
    from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain.retrievers.document_compressors import CrossEncoderReranker
    ADVANCED_RETRIEVERS_AVAILABLE = True
except ImportError:
    print("Warning: Advanced retrievers not available, using basic dense retriever")
    ADVANCED_RETRIEVERS_AVAILABLE = False

def load_llama_model(model_id="meta-llama/Llama-3.1-8B-Instruct",  max_new_tokens=512, temperature=0.1):
    """Llama 모델 & LangChain Pipeline 로드"""
    load_dotenv(override=True)
    HF_TOKEN = os.environ.get("HF_TOKEN", "")
    login(token=HF_TOKEN)   

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    model.to(device)

    text_generation_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        top_p=0.95,
        temperature=temperature,
        device=0 if device == "cuda" else -1
    )

    llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
    return llm

def load_embedding_model(model_name="BAAI/bge-base-en-v1.5", device='cuda'):
    """Embedding 모델 로드"""
    embeddings =HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device}, 
        encode_kwargs={'normalize_embeddings': True}
    )
    return embeddings

def load_datasets(mode:str = 'distractor'):
    if mode == 'distractor':
        data_path = "/mnt/aix23604/hotpotqa/distractor"
        dataset = load_from_disk(data_path)
    else : 
        data_path = "/mnt/aix23604/hotpotqa/fullwiki"
        dataset = load_from_disk(data_path)
    return dataset


def sparse_retriever(retriever_k, df_corpus):
    bm25_retriever = BM25Retriever.from_documents(
        [Document(page_content=text, metadata={"title": title})
        for text, title in zip(df_corpus['text'], df_corpus['title'])])
    bm25_retriever.k = retriever_k
    return bm25_retriever


def dense_retriever(retriever_k, vectorstore):
    dense_ret = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    return dense_ret


def hybrid_retriever(retriever_k, df_corpus, vectorstore, sparse_wt=0.5, dense_wt=0.5):
    bm25_ret = sparse_retriever(retriever_k, df_corpus)
    dense_ret = dense_retriever(retriever_k, vectorstore)
    
    if ADVANCED_RETRIEVERS_AVAILABLE:
        hybrid_ret = EnsembleRetriever(
            retrievers=[bm25_ret, dense_ret],
            weights=[sparse_wt, dense_wt]
        )
        return hybrid_ret
    else:
        # Fallback: dense만 사용
        print("Using dense retriever only (hybrid not available)")
        return dense_ret


def cross_encoder(retriever_k, cross_encoder_k, df_corpus, vectorstore, emb_model="BAAI/bge-base-en-v1.5"):
    base_retriever = hybrid_retriever(retriever_k, df_corpus, vectorstore)
    
    if ADVANCED_RETRIEVERS_AVAILABLE:
        model = HuggingFaceCrossEncoder(model_name=emb_model)
        compressor = CrossEncoderReranker(model=model, top_n=cross_encoder_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )
        return compression_retriever
    else:
        # Fallback: base retriever만 사용
        print("Using base retriever only (cross-encoder not available)")
        return base_retriever