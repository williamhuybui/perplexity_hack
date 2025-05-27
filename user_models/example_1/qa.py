from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import RetrievalQA
import pandas as pd
import os, json, csv

PDF_FOLDER = "./data"
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500
VECTOR_STORE_DIR = "chroma_index_finance"
LOG_FILE = "qa_log_1.csv"

def create_chain():

    # === Load all PDFs ===
    all_documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, filename)
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = filename  # track which PDF it came from
            all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} total documents.")

    # === Split text into chunks ===
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(all_documents)

    # Add unique chunk IDs and ensure source is in metadata
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        doc.metadata["source"] = doc.metadata.get("source", "unknown")

    #Embbedding
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False} #False Euclidean, True cosine similarity
    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    #Vector Store
    vector_store = FAISS.from_documents(chunks, hf)
    vector_store.save_local("faiss_index_open")

    #Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    #LLM
    llm = ChatPerplexity(
        model="sonar",
        pplx_api_key = "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",
        temperature=0.2
    )

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )
    
    return qa_chain

QA1 = create_chain()