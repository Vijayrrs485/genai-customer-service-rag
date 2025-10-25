from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from utils.document_loader import load_documents, split_documents
import config
import os

def create_vector_store(embeddings):
    """Create FAISS vector store from documents"""
    print("Loading documents...")
    documents = load_documents()
    
    print(f"Loaded {len(documents)} documents")
    print("Splitting documents...")
    chunks = split_documents(documents)
    
    print(f"Created {len(chunks)} chunks")
    print("Creating vector store...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    # Save vector store
    vector_store.save_local(config.FAISS_INDEX_PATH)
    print(f"Vector store saved to {config.FAISS_INDEX_PATH}")
    
    return vector_store

def load_vector_store(embeddings):
    """Load existing FAISS vector store"""
    if os.path.exists(config.FAISS_INDEX_PATH):
        vector_store = FAISS.load_local(
            config.FAISS_INDEX_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )
        return vector_store
    else:
        print("No existing index found. Creating new one...")
        return create_vector_store(embeddings)