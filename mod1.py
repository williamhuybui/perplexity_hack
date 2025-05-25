from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import RetrievalQA

#Loader
loader = PyMuPDFLoader("data\Huy_Bui_Resume.pdf")
documents = loader.load()

#Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

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
    model="sonar-pro",
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