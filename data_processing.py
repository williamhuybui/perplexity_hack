from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from llm import llm_summary_extraction
from utils import load_metadata
import os, json

def pdf_load_and_split(pdf_path, cfg):
    """ 
    Convert a PDF file to chunks of text.
    Args:
        pdf_path (str): Path to the PDF file.
        cfg (Config): Configuration object containing chunk size and overlap.
    Returns:
        list: List of text chunks.
        documents: List of Document objects containing page content.
    """
    loader = PyMuPDFLoader(str(pdf_path))
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap
    )
    chunks = splitter.split_documents(documents)
    return [c.page_content for c in chunks], documents


# Loop through PDFs
def data_processing(api_key, cfg, verbose=True):
    """ 
    Process PDF files in the input directory, extract summaries, and save metadata.
    This function performs the following steps:
    1. Load metadata from a JSON file.
    2. For each PDF file in the input directory:
        a. Load and split the PDF into chunks.
        b. Extract a summary from the first few pages.
        c. Save metadata including file name, format name, file path, chunk count, total word count, and summary.
        d. Save the chunks to a JSON file.
        e. Save metadata to a JSON file.

    Args:
        cfg (Config): Configuration object containing API key and number of questions.
        api_key (str): API key for the LLM service.
    Returns:
        CSV: with questions and their corresponding sources.
    """

    all_filenames = [file_name for file_name in os.listdir(cfg.input_dir) if file_name.endswith(".pdf")]
    metadata = load_metadata(cfg)

    for i, file_name in enumerate(all_filenames):
        if file_name not in metadata:
            print(f"Processing file: {file_name} {i+1}/{len(all_filenames)}")
            file_path = os.path.join(cfg.input_dir, file_name)

            # 1) Load and split PDF
            chunks, documents = pdf_load_and_split(pdf_path=file_path, cfg=cfg)
            
            # 2) Summary
            first_n_pages = "\n".join([doc.page_content for doc in documents[:cfg.n_page_summary]])
            summary = llm_summary_extraction(api_key = api_key, first_n_pages = first_n_pages)        
            
            # 3) Save metadata
            format_name = file_name.split(".")[0]
            metadata[file_name] = {
                "format_name": format_name,
                "file_path": file_path,
                "chunk_count": len(chunks),
                "total_word_count": sum(len(chunk.split()) for chunk in chunks),
                "summary": summary,
            }

            # 4) Save chunks
            file_chunks_dir = os.path.join(cfg.chunks_dir, f"{format_name}.json")
            with open(file_chunks_dir, "w") as f:
                if verbose:
                    print(f"Saving {len(chunks)} chunks to {file_chunks_dir}")
                json.dump(chunks, f, indent=2)

            # Save metadata to file
            with open(cfg.metadata_file, "w") as f:
                if verbose:
                    print(f"Saving metadata to {cfg.metadata_file}")
                json.dump(metadata, f, indent=2)
        else:
            if verbose:
                print(f"File '{file_name}' already processed. Skipping.")