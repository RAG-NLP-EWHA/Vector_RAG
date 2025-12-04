import os
import pandas as pd
import evaluate as hf_evaluate
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from utils import load_llama_model, load_embedding_model

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["RAGAS_TIMEOUT"] = "300" 


OUTPUT_PATH = "/mnt/aix23604/output"
RESULT_PATH = os.path.join(OUTPUT_PATH, 'llama_sample_validation.json')
EMB_MODEL = "BAAI/bge-base-en-v1.5" 

# 평가용 Llama 모델과 임베딩 모델 로드
judge_llm = load_llama_model()
judge_embeddings = load_embedding_model()

# 결과 불러오기
df_results = pd.read_json(RESULT_PATH)
df_results_sample = df_results.head(1).copy()

# 답변 정리
def clean_answer(answer):
    if "Answer:" in answer:
        answer = answer.split("Answer:")[-1].strip()
    if "Human:" in answer:
        answer = answer.split("Human:")[0].strip()
    return answer.strip()

df_results_sample['answer'] = df_results_sample['answer'].apply(clean_answer)

## EM, F1 산출 - EM의 경우 문자열 정리 코드 추가 필요 
squad_metric = hf_evaluate.load("squad")

predictions = [{"prediction_text": row['answer'], "id": str(i)} for i, row in df_results_sample.iterrows()]
references = [{"answers": {"answer_start": [0], "text": [row['ground_truth']]}, "id": str(i)} for i, row in df_results_sample.iterrows()]
results_squad = squad_metric.compute(predictions=predictions, references=references)
print("SQuAD Metrics:", results_squad)

## RAGAS 평가 
print("RAGAS Evaluation Starting")
ragas_dataset = Dataset.from_pandas(df_results_sample)
print("RAGAS Dataset:", ragas_dataset)

print("Evaluating with RAGAS")
ragas_results = ragas_evaluate(
    ragas_dataset,
    # metrics=[
    #     context_precision,
    #     context_recall,
    #     faithfulness,
    #     answer_relevancy,
    # ],
    metrics = [faithfulness],
    llm=judge_llm,
    embeddings=judge_embeddings
)

df_ragas = ragas_results.to_pandas()
df_ragas.to_csv(os.path.join(OUTPUT_PATH,'llama_validation_evaluation_wrapping.csv'), index=False, encoding='utf-8-sig')
print(f"Evaluation results saved to {OUTPUT_PATH}/llama_validation_evaluation_wrapping.csv")