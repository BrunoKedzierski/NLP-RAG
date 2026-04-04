from database import load_database
from generate import generate_response, get_llm
from retriever import get_retriever
from loguru import logger
import argparse



def main(hybrid_search,llm_model, verbose):
    logger.info("Loading Chroma database...")
    db =  load_database(db_name="chroma_db_v1")
    retriever = get_retriever(db, hybrid_search)
    logger.info("Database loaded.")
    logger.info("Initializing llm...")
    llm = get_llm(model_name=llm_model)

    while True:
        # Prompt user for input
        query = input("\nEnter your query (or type 'exit' to quit): ").strip()
        if query.lower() in {"exit", "quit"}:
            logger.info("Exiting program. Goodbye!")
            break

        if not query:
            print("Please enter a non-empty query.")
            continue

        try:
            logger.info("Retrieving documents")
            chunks = retriever.invoke(query)
            if verbose:
                print("\n=== Retrieved Chunks (with metadata) ===")
                for i, doc in enumerate(chunks, 1):
                    print(f"\n--- Chunk {i} ---")
                    print("Content:", doc.page_content)
                    print("Metadata:", doc.metadata)
            logger.info("Generating response")
            response = generate_response(llm,chunks, query)
            print("\n=== Response ===")
            print(response)
        except Exception as e:
            logger.error(f"Error processing query: {e}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Query the Chroma database with optional hybrid search.")
    parser.add_argument("--hybrid-search", action="store_true", help="Enable hybrid search (dense + sparse retrieval)")
    parser.add_argument("--verbose", action="store_true", help="Enable hybrid search (dense + sparse retrieval)")
    parser.add_argument("--llm", type=str, default="gemma3:4b", required=False,help="llm model name")

    args = parser.parse_args()

    main(hybrid_search=args.hybrid_search, llm_model = args.llm, verbose = args.verbose)