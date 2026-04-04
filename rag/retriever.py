from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document



def get_retriever(chroma_db, hybrid_search = False):
     dense_retriever = chroma_db.as_retriever(search_kwargs={"k": 3})
     if not hybrid_search:
          return dense_retriever
     raw_docs = chroma_db.get(include=["documents", "metadatas"]) # Convert them in Document object 
     documents = [ Document(page_content=doc, metadata=meta) for doc, meta in zip(raw_docs["documents"], raw_docs["metadatas"]) ] 
     bm25_retriever = BM25Retriever.from_documents(documents=documents, k=3) 
     ensemble_retriever = EnsembleRetriever(retrievers=[dense_retriever, bm25_retriever], weights=[0.7, 0.5])
     return ensemble_retriever