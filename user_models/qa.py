from langchain.document_loaders import PyMuPDFLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS, Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatPerplexity
import os


MODEL_1_CONFIG = {
    "model_name": "model_1",
    "model_type": "faiss",
    
    # Data paths
    "pdf_folder": "./data",
    "vector_store_path": "faiss_index_open",
    "questions_file": "session_1/questions.csv",
    "log_file": "qa_log_1.csv",
    "final_output_file": "final_1.csv",
    
    # Text processing
    "chunk_size": 5000,
    "chunk_overlap": 500,
    "text_splitter_type": "recursive",  # "recursive" or "token"
    "document_loader": "pymupdf",  # "pymupdf" or "pypdf"
    
    # Embeddings
    "embedding_model": "sentence-transformers/all-mpnet-base-v2",
    "embedding_type": "huggingface",
    "embedding_device": "cpu",
    "normalize_embeddings": False,
    
    # LLM settings
    "llm_model": "sonar",
    "pplx_api_key": "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",
    "temperature": 0.2,
    
    # Retrieval settings
    "retriever_k": 3,
    "use_compression": False,
    "use_custom_prompt": False,
    
    # Processing settings
    "recreate_vector_store": False,
    "max_retries": 3
}

def create_vector_store(config = MODEL_1_CONFIG):
    """Create vector store based on configuration"""
    
    # === Load all PDFs ===
    all_documents = []
    loader_class = PyMuPDFLoader if config["document_loader"] == "pymupdf" else PyPDFLoader
    
    for filename in os.listdir(config["pdf_folder"]):
        if filename.endswith(".pdf"):
            file_path = os.path.join(config["pdf_folder"], filename)
            loader = loader_class(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = filename
            all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} total documents.")

    # === Split text into chunks ===
    if config["text_splitter_type"] == "token":
        text_splitter = TokenTextSplitter(
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["chunk_overlap"]
        )
    else:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["chunk_size"], 
            chunk_overlap=config["chunk_overlap"]
        )
    
    chunks = text_splitter.split_documents(all_documents)

    # Add unique chunk IDs and ensure source is in metadata
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        doc.metadata["source"] = doc.metadata.get("source", "unknown")

    print(f"Created {len(chunks)} text chunks.")

    # === Create embeddings ===
    if config["embedding_type"] == "bge":
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=config["embedding_model"],
            model_kwargs={'device': config["embedding_device"]},
            encode_kwargs={'normalize_embeddings': config["normalize_embeddings"]}
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding_model"],
            model_kwargs={'device': config["embedding_device"]},
            encode_kwargs={'normalize_embeddings': config["normalize_embeddings"]}
        )

    # === Create vector store ===
    if config["model_type"] == "chroma":
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=config["vector_store_path"]
        )
    else:  # FAISS
        vector_store = FAISS.from_documents(chunks, embeddings)
        vector_store.save_local(config["vector_store_path"])
    
    print(f"Vector store created and saved to: {config['vector_store_path']}")
    return vector_store

def load_vector_store(config = MODEL_1_CONFIG):
    """Load existing vector store based on configuration"""
    
    # Create embeddings (must match the ones used to create the store)
    if config["embedding_type"] == "bge":
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=config["embedding_model"],
            model_kwargs={'device': config["embedding_device"]},
            encode_kwargs={'normalize_embeddings': config["normalize_embeddings"]}
        )
    else:
        embeddings = HuggingFaceEmbeddings(
            model_name=config["embedding_model"],
            model_kwargs={'device': config["embedding_device"]},
            encode_kwargs={'normalize_embeddings': config["normalize_embeddings"]}
        )
    
    # Load vector store
    if config["model_type"] == "chroma":
        vector_store = Chroma(
            persist_directory=config["vector_store_path"],
            embedding_function=embeddings
        )
    else:  # FAISS
        vector_store = FAISS.load_local(
            config["vector_store_path"], 
            embeddings, 
            allow_dangerous_deserialization=True
        )
    
    print(f"Vector store loaded from: {config['vector_store_path']}")
    return vector_store

def create_unified_chain(config = MODEL_1_CONFIG):
    """Create QA chain based on unified configuration"""
    
    # === Handle vector store ===
    vector_store_exists = (
        os.path.exists(config["vector_store_path"]) if config["model_type"] == "chroma"
        else os.path.exists(config["vector_store_path"])
    )
    
    if config["recreate_vector_store"] or not vector_store_exists:
        print(f"Creating new vector store for {config['model_name']}...")
        vector_store = create_vector_store(config)
    else:
        print(f"Loading existing vector store for {config['model_name']}...")
        vector_store = load_vector_store(config)

    # === Create base retriever ===
    base_retriever = vector_store.as_retriever(
        search_kwargs={"k": config["retriever_k"]}
    )

    # === Initialize LLM ===
    llm = ChatPerplexity(
        model=config["llm_model"],
        pplx_api_key=config["pplx_api_key"],
        temperature=config["temperature"]
    )

    # === Setup retriever (with or without compression) ===
    if config["use_compression"]:
        compressor = LLMChainExtractor.from_llm(llm)
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever
        )
        print("Using compression retriever")
    else:
        retriever = base_retriever
        print("Using standard retriever")

    # === Setup prompt (if custom prompt specified) ===
    chain_type_kwargs = {}
    if config["use_custom_prompt"]:
        prompt = PromptTemplate(
            template=config["custom_prompt"],
            input_variables=["context", "question"]
        )
        chain_type_kwargs = {"prompt": prompt}
        print("Using custom prompt template")

    # === Create QA chain ===
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs=chain_type_kwargs
    )
    
    print(f"QA chain created successfully for {config['model_name']}")
    return qa_chain