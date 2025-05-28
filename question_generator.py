from pathlib import Path
import json, random
from llm import llm_generate_questions
from utils import load_metadata
import pandas as pd
import os

def load_questions(cfg) :
    """
    Load questions from a CSV file.
    Args:
        cfg (Config): Configuration object containing questions file path.
    Returns:
        pd.DataFrame: DataFrame containing questions and their corresponding metadata.
    """
    if os.path.exists(cfg.questions_file):
        df = pd.read_csv(cfg.questions_file)
        return df
    else:
        print(f"Questions file '{cfg.questions_file}' does not exist.")
        return pd.DataFrame()
    

def question_generator(cfg, verbose = True):
    """
    For each PDF in *metadata* pick `cfg.n_questions_per_file` random chunks,
    generate a question via `llm_generate_questions`, and return
    all rows as a DataFrame.
    Args:
        cfg (Config): Configuration object containing parameters.
        metadata (list): List of metadata entries for each PDF.
    Returns:
        pd.DataFrame: DataFrame containing questions and their corresponding metadata.
    """
    rows = []
    metadata = load_metadata(cfg)

    if os.path.exists(cfg.question_file):
        if verbose:
            print(f"Questions file '{cfg.question_file}' already exists. Loading existing questions.")
        return pd.read_csv(cfg.question_file)

    for i, file_name in enumerate(metadata):
        if verbose:
            print(f"Generating {cfg.n_questions_per_file} question for file: {file_name}. {i+1}/{len(metadata)}")
        meta = metadata[file_name]
        format_name = meta["format_name"]
        file_path = meta["file_path"]
        summary = meta["summary"]

        for _ in range(cfg.n_questions_per_file):
            # Randomly select a chunk from the file
            chunk_path = Path(cfg.chunks_dir) / f"{format_name}.json"
            with chunk_path.open() as f:
                chunks = json.load(f)

            chunk_id = random.randrange(len(chunks))
            chunk = chunks[chunk_id]
            question = llm_generate_questions(cfg.api_key, summary,chunk)
            rows.append(
                {
                    "file_name": file_name,
                    "question": question,
                    "format_name": format_name,
                    "file_path": file_path,
                    "summary": summary,
                    "chunk": chunk,
                    "chunk_id": chunk_id,
                }
            )
    df = pd.DataFrame.from_records(rows)
    df.to_csv(cfg.question_file, index=False)
    if verbose:
        print(f"Questions saved to {cfg.question_file}")
    return df
