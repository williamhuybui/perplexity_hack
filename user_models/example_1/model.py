from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import RetrievalQA
import os

def model():
    # Settings
    PDF_FOLDER = "/Users/huybui/Desktop/SIMPLE_RAG/data"
    CHUNK_SIZE = 5000
    CHUNK_OVERLAP = 500

    # Load all PDFs
    all_documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.endswith(".pdf"):
            file_path = os.path.join(PDF_FOLDER, filename)
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
            all_documents.extend(documents)
    print(f"Loaded {len(all_documents)} total documents.")

    # Split
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(all_documents)

    print(f"Split into {len(chunks)} chunks.")

    # Embedding
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}  # False = Euclidean, True = Cosine similarity

    hf = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    # Vector Store
    vector_store = FAISS.from_documents(chunks, hf)
    # vector_store.save_local("faiss_index_open")
    print("Vector store saved.")

    # Retriever
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # LLM setup
    llm = ChatPerplexity(
        model="sonar",
        pplx_api_key="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",
        temperature=0.2
    )

    # QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True
    )

    print("QA chain is ready.")
    return qa_chain