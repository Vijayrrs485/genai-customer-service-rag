from langchain.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import config
import os

def load_documents(directory_path="data/technical_docs"):
    """Load documents from directory"""
    documents = []
    
    # Load PDF documents
    pdf_loader = DirectoryLoader(
        directory_path,
        glob="/*.pdf",
        loader_cls=PyPDFLoader
    )
    documents.extend(pdf_loader.load())
    
    # Load text documents
    txt_loader = DirectoryLoader(
        directory_path,
        glob="/*.txt",
        loader_cls=TextLoader
    )
    documents.extend(txt_loader.load())
    
    return documents

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks