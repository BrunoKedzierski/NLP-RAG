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




PROJECT_ROOT = Path(__file__).resolve().parent.parent
TESTSET_FOLDER = PROJECT_ROOT / "testset"

import json



# def chunk_list(lst, size):
#     for i in range(0, len(lst), size):
#         yield lst[i:i + size]


# def generate_response(llm, retrieved_chunks):
       
#     try:
#             # Build the text prompt
#         prompt_text = f"""Based on the following documents, Generate:
#                     1. 15 realistic user questions
#                     2. A precise answers for each one,  based only on context

#                     Return JSON.

# CONTENT TO ANALYZE:
# """
        
#         for i, chunk in enumerate(retrieved_chunks):
#             prompt_text += f"--- Document {i+1} ---\n"
            
#             if "original_content" in chunk.metadata:
#                 original_data = json.loads(chunk.metadata["original_content"])
                
#                 # Add raw text
#                 raw_text = original_data.get("raw_text", "")
#                 if raw_text:
#                     prompt_text += f"TEXT:\n{raw_text}\n\n"
                
#                 # Add tables as HTML
#                 tables_html = original_data.get("tables_html", [])
#                 if tables_html:
#                     prompt_text += "TABLES:\n"
#                     for j, table in enumerate(tables_html):
#                         prompt_text += f"Table {j+1}:\n{table}\n\n"
            
#             prompt_text += "\n"
        

#         # Build message content starting with text
#         message_content = [{"type": "text", "text": prompt_text}]
        
#         # Add all images from all chunks
#         for chunk in retrieved_chunks:
#             if "original_content" in chunk.metadata:
#                 original_data = json.loads(chunk.metadata["original_content"])
#                 images_base64 = original_data.get("images_base64", [])
                
#                 for image_base64 in images_base64:
#                     message_content.append({
#                         "type": "image_url",
#                         "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
#                     })
        
#         # Send to AI and get response
#         message = HumanMessage(content=message_content)
#         response = llm.invoke([message])
        
#         return response.content
            
#     except Exception as e:
#         print(f"❌ Answer generation failed: {e}")
#         return "Sorry, I encountered an error while generating the answer."





# def generate_test_set():
#     llm_generator = get_llm("qwen2.5vl:3b")
#     db = load_database(db_name="chroma_db_v1")

#     docs = db.similarity_search("", k=9)

#     results = []

#     for chunk in chunk_list(docs, 3):
#         response = generate_response(llm_generator, chunk)
#         results.append(response)

#     print("\n\n".join(results))


def load_test_set():
 with open(f"{TESTSET_FOLDER}/testset.json", mode="r", encoding="utf-8") as read_file:
     test_set = json.load(read_file)
     return test_set

def run_eval():

    test_set = load_test_set()

    db = load_database(db_name= "chroma_db_v1")
    retriever = get_retriever(db)
    llm = get_llm()

    print("initialized llm")

    dataset = []

    for entry in test_set:
        print("processing question....")
        relevant_docs = retriever.invoke(entry["query"])
        print("retrieved context....")
        response = generate_response(llm,relevant_docs, entry["query"])
        print("generated_rsponse....")
        dataset.append(
            {
                "user_input": entry["query"],
                "retrieved_contexts": [rdoc.page_content for rdoc in relevant_docs],
                "response": response,
                "reference": entry["expected_response"],
            }
        )

    print(dataset)

    evaluation_dataset = EvaluationDataset.from_list(dataset)
    evaluator_llm = LangchainLLMWrapper(llm)
    print("evaluating....")
    result = evaluate(
        dataset=evaluation_dataset,
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness()],
        llm=evaluator_llm,
        run_config=RunConfig(max_workers=1)
    )

    print(result)

if __name__ == "__main__":
    run_eval()
    