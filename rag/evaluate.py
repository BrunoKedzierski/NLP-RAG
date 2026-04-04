from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from pathlib import Path
from database import load_database
from generate import generate_response, get_llm
from retriever import get_retriever
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTSET_FOLDER = PROJECT_ROOT / "testset"

import json


def load_test_set():
 with open(f"{TESTSET_FOLDER}/testset.json", mode="r", encoding="utf-8") as read_file:
     test_set = json.load(read_file)
     return test_set

def run_eval():

    test_set = load_test_set()

    db = load_database(db_name= "chroma_db_v1")
    retriever = get_retriever(db)
    llm = get_llm()


    dataset = []

    for entry in test_set:
        relevant_docs = retriever.invoke(entry["query"])
        response = generate_response(llm,relevant_docs, entry["query"])
        dataset.append(
            {
                "user_input": entry["query"],
                "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
                "response": response,
                "reference": entry["expected_response"],
            }
        )

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(llm)
    
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
    )

    print(result)

if __name__ == "__main__":
   run_eval()