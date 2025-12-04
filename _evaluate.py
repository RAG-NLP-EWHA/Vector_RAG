'''
openAI env 추가 필요 
'''
OUTPUT_PATH = "/mnt/aix23604/output"
RESULT_PATH = os.path.join(OUTPUT_PATH, 'sample.json')
judge_llm_model="gpt-4o-mini"

import pandas as pd
import evaluate
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

df_results = pd.read_json(RESULT_PATH)


## EM, F1 산출 - EM의 경우 문자열 정리 코드 추가 필요 
squad_metric = evaluate.load("squad")

predictions = [{"prediction_text": row['answer'], "id": str(i)} for i, row in df_results.iterrows()]
references = [{"answers": {"answer_start": [0], "text": [row['ground_truth']]}, "id": str(i)} for i, row in df_results.iterrows()]
results_squad = squad_metric.compute(predictions=predictions, references=references)
print(result_squad)

## RAGAS 평가 
ragas_dataset = Dataset.from_pandas(df_results)
print(ragas_dataset)

judge_llm = ChatOpenAI(model=judge_llm_model, temperature=0)
judge_embeddings = OpenAIEmbeddings()

ragas_results = ragas_evaluate(
    ragas_dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
    llm=judge_llm,
    embeddings=judge_embeddings
)

df_ragas = ragas_results.to_pandas()
df_ragas.to_csv(os.path.join(OUTPUT_PATH,'evaluation_result_samples.csv'), index=False, encoding='utf-8-sig')
