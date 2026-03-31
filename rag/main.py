from database import load_database


def main():
    db = load_database(db_name="chroma_db_v1")
    query = "What are the two main components of the Transformer architecture? "
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chunks = retriever.invoke(query)
    print(chunks[0])


if __name__ == "__main__":
    main()
