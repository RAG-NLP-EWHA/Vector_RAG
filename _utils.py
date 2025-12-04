from datasets import load_from_disk

def load_datasets(mode:str = 'distractor'):
    if mode == 'distractor':
        data_path = "/mnt/aix23604/hotpotqa/distractor"
        dataset = load_from_disk(data_path)
        
    else : 
        data_path = "/mnt/aix23604/hotpotqa/fullwiki"
        dataset = load_from_disk(data_path)
        
    return dataset


def sparse_retriever(retriever_k):
    bm25_retriever = BM25Retriever.from_documents(
        [Document(page_content=text, metadata={"title": title})
        for text, title in zip(df_corpus['text'], df_corpus['title'])])
    
    bm25_retriever.k = retriever_k
    
    return bm25_retriever

def dense_retriever(retriever_k):
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": retriever_k})
    
    return dense_retriever

def hybrid_retriever(retriever_k, sparse_wt=0.5, dense_wt=0.5):
    bm25_retriever = sparse_retriever(retriever_k)
    dense_retriever = dense_retriever(retriever_k)
    
    hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights = [sparse_wt, dense_wt])
    
    return hybrid_retriever

def cross_encoder(emb_model, cross_encoder_k):
    model = HuggingFaceCrossEncoder(model_name=emb_model)
    compressor = CrossEncoderReranker(model=model, top_n=cross_encoder_k) # 상위 k개만 최종 선택

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=hybrid_retriever)
    return compression_retriever