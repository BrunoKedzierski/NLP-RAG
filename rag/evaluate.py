from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset
from pathlib import Path
from database import load_database
from generate import generate_response, get_llm
from retriever import get_retriever
from ragas import evaluate
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness
from langchain_core.documents import Document
import random
from ragas.testset import TestsetGenerator
from langchain_ollama import OllamaEmbeddings
from langchain_core.messages import HumanMessage
from itertools import islice
from ragas.run_config import RunConfig
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTSET_FOLDER = PROJECT_ROOT / "testset"
RESULTS_FOLDER = TESTSET_FOLDER / "results"
GENERATIONS_FOLDER = TESTSET_FOLDER / "generations"

import json
from datetime import datetime


def load_test_set():
    with open(
        f"{TESTSET_FOLDER}/testset.json", mode="r", encoding="utf-8"
    ) as read_file:
        test_set = json.load(read_file)
        return test_set


def run_eval(
    generate_model="gemma3:4b", evaluator_model="gemma3:4b", hybrid_search=False
):

    # test_set = load_test_set()
    test_set = random.sample(load_test_set(), k=10)

    db = load_database(db_name="chroma_db_v1")
    retriever = get_retriever(db, hybrid_search=hybrid_search)
    llm = get_llm(model_name=generate_model)
    evaluator_llm = get_llm(model_name=evaluator_model)

    print("initialized llm")

    GENERATIONS_FOLDER.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = (
        f"gen-{generate_model.replace(':', '_')}"
        f"_eval-{evaluator_model.replace(':', '_')}"
        f"_hybrid-{hybrid_search}"
    )
    generations_path = GENERATIONS_FOLDER / f"{timestamp}_{run_tag}.json"

    dataset = []

    for entry in tqdm(test_set, desc="Evaluating"):
        relevant_docs = retriever.invoke(entry["query"])
        response = generate_response(llm, relevant_docs, entry["query"])
        dataset.append(
            {
                "user_input": entry["query"],
                "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
                "response": response,
                "reference": entry["expected_response"],
            }
        )
    with open(generations_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "generate_model": generate_model,
                "evaluator_model": evaluator_model,
                "hybrid_search": hybrid_search,
                "timestamp": timestamp,
                "samples": dataset,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved generations to {generations_path}")

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(evaluator_llm)
    print("evaluating....")
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
        run_config=RunConfig(max_workers=2),
    )

    # print(result)

    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_FOLDER / f"{timestamp}_{run_tag}.json"

    result_df = result.to_pandas()
    summary = {k: float(v) for k, v in result_df.select_dtypes("number").mean().items()}
    payload = {
        "generate_model": generate_model,
        "evaluator_model": evaluator_model,
        "hybrid_search": hybrid_search,
        "timestamp": timestamp,
        "summary": summary,
        "samples": result_df.to_dict(orient="records"),
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    run_eval(
        generate_model="gemma3:4b", evaluator_model="gemma3:4b", hybrid_search=False
    )
    run_eval(
        generate_model="gemma3:4b", evaluator_model="gemma3:4b", hybrid_search=True
    )
    run_eval(
        generate_model="qwen3.5:4b", evaluator_model="gemma3:4b", hybrid_search=False
    )
    run_eval(
        generate_model="qwen3.5:4b", evaluator_model="gemma3:4b", hybrid_search=True
    )
