from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatPerplexity
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
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
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            for doc in documents:
                doc.metadata["source"] = filename  # track which PDF it came from
            all_documents.extend(documents)

    print(f"Loaded {len(all_documents)} total documents.")

    # === Split text into chunks ===
    text_splitter = TokenTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = text_splitter.split_documents(all_documents)

    # Add unique chunk IDs and ensure source is in metadata
    for i, doc in enumerate(chunks):
        doc.metadata["chunk_id"] = i
        doc.metadata["source"] = doc.metadata.get("source", "unknown")

    # === Embedding model ===
    model_name = "BAAI/bge-base-en"
    hf = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    # === Vector Store ===
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=hf,
        persist_directory=VECTOR_STORE_DIR
    )
    # vector_store.persist()

    # === Prompt Template (only answer output) ===
    prompt_template = """
    You are a professional financial advisor with expertise in corporate finance, investment analysis, and career development in finance-related roles.

    Use only the information provided in the context to answer the user's question.
    Do not make assumptions or fabricate any details.

    Respond clearly and professionally, as if advising a client on their financial career or investment decisions.

    {context}

    Question: {question}

    If the answer is not explicitly stated in the context, respond with: "I don't know based on the provided document".
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # === Retriever Setup ===
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    # Perplexity LLM
    perplexity_llm = ChatPerplexity(
        model="sonar",
        pplx_api_key="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",
        temperature=0.2
    )

    # Compression retriever
    compressor = LLMChainExtractor.from_llm(perplexity_llm)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )

    # === QA Chain ===
    qa_chain = RetrievalQA.from_chain_type(
        llm=perplexity_llm,
        chain_type="stuff",
        retriever=compression_retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain

QA2 = create_chain()