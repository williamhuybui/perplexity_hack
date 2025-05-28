import pandas as pd
import os, json, csv
from typing import Optional, List, Dict, Any

def process_answer(qa_chain, query: str, log_file: str = "qa_log_1.csv") -> Dict[str, Any]:
    """
    Process a single question through the QA chain
    
    Args:
        qa_chain: The QA chain object
        query: Question to process
        log_file: Path to log CSV file
        
    Returns:
        Dictionary with processed answer and metadata
    """
    response = qa_chain({"query": query})
    
    # Handle case where result is a JSON-formatted string
    try:
        result = json.loads(response["result"])
        answer_text = result.get("answer", response["result"])
    except (json.JSONDecodeError, TypeError):
        # Fallback: use the raw string if not JSON
        answer_text = response["result"]
    
    source_docs = response['source_documents']
    
    sources_info = []
    for doc in source_docs:
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        source = doc.metadata.get("source", "N/A")
        sources_info.append({"chunk_id": str(chunk_id), "source": source})
    
    # Prepare data to log
    top_k_chunks = [src["chunk_id"] for src in sources_info if src["chunk_id"] != "N/A"]
    sources_list = [src["source"] for src in sources_info if src["source"] != "N/A"]
    
    # Write to CSV
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "sources", "top_k_chunks"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "question": query,
            "answer": answer_text,
            "sources": "; ".join(list(set(sources_list))),
            "top_k_chunks": "; ".join(list(set(top_k_chunks)))
        })
    
    return {
        "question": query,
        "answer": answer_text,
        "sources": sources_list,
        "top_k_chunks": top_k_chunks,
        "sources_info": sources_info
    }

def qa(qa_chain, 
       questions_file: str = "session_1/questions.csv",
       log_file: str = "qa_log_1.csv") -> pd.DataFrame:
    """
    Process all questions from CSV file through QA chain
    
    Args:
        qa_chain: The QA chain object
        questions_file: Path to questions CSV file
        log_file: Path to log CSV file
        
    Returns:
        DataFrame with all processed Q&A results
    """
    # Clear existing log file if it exists
    if os.path.exists(log_file):
        os.remove(log_file)
        print(f"Cleared existing log file: {log_file}")
    
    questions = pd.read_csv(questions_file)
    print(f"Processing {len(questions)} questions from {questions_file}")
    
    for index, row in questions.iterrows():
        question = row['question']
        print(f"Processing question {index + 1}/{len(questions)}: {question[:50]}...")
        process_answer(qa_chain, question, log_file)
    
    df = pd.read_csv(log_file)
    print(f"Completed processing. Results saved to {log_file}")
    return df

def finalize(evaluate_answer_func, 
            df: pd.DataFrame,
            final_output_file: str = "final_1.csv",
            max_retries: int = 3) -> pd.DataFrame:
    """
    Finalize results by adding evaluation scores
    
    Args:
        evaluate_answer_func: Function to evaluate answers
        df: DataFrame with Q&A results
        final_output_file: Path to save final results
        max_retries: Maximum number of retries for failed evaluations
        
    Returns:
        DataFrame with evaluation results added
    """
    final_rows = []
    
    for index, row in df.iterrows():
        question = row['question']
        top_k_chunk = row['top_k_chunks']
        answer = row['answer']
        
        print(f"Evaluating answer {index + 1}/{len(df)}: {question[:50]}...")
        
        success = False
        retry_count = 0
        evaluation = {}
        
        while not success and retry_count < max_retries:
            try:
                evaluation = evaluate_answer_func(question, top_k_chunk, answer)
                success = True  # Break loop if successful
            except Exception as e:
                retry_count += 1
                print(f"Retry {retry_count}/{max_retries} for question: {question[:50]}... due to error: {e}")
                if retry_count >= max_retries:
                    print(f"Failed to evaluate after {max_retries} retries. Using default values.")
                    evaluation = {"error": str(e)}
        
        # Build a combined result dictionary
        result_row = {
            'question': question,
            'top_k_chunk': top_k_chunk,
            'answer': answer
        }
        
        # Add evaluation results
        for key, value in evaluation.items():
            result_row[f'evaluation_{key}'] = value
        
        final_rows.append(result_row)
    
    # Convert list of results to DataFrame
    final_df = pd.DataFrame(final_rows)
    
    # Save to CSV
    final_df.to_csv(final_output_file, index=False)
    print(f"Saved final results to {final_output_file}")
    return final_df

def run_qa_pipeline(qa_chain,
                   questions_file: str = "session_1/questions.csv",
                   log_file: str = "qa_log_1.csv",
                   final_output_file: str = "final_1.csv",
                   evaluate_answer_func = None,
                   max_retries: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run the complete QA pipeline
    
    Args:
        qa_chain: The QA chain object
        questions_file: Path to questions CSV file
        log_file: Path to log CSV file
        final_output_file: Path to save final results
        evaluate_answer_func: Function to evaluate answers (optional)
        max_retries: Maximum retries for evaluations
        
    Returns:
        Tuple of (qa_results_df, final_results_df)
    """
    print("=== Starting QA Pipeline ===")
    
    # Step 1: Process all questions
    df = qa(qa_chain, questions_file, log_file)
    
    # Step 2: Finalize with evaluations (if evaluation function provided)
    if evaluate_answer_func:
        print("=== Starting Evaluation ===")
        final_df = finalize(evaluate_answer_func, df, final_output_file, max_retries)
        return df, final_df
    else:
        print("=== No evaluation function provided, skipping evaluation ===")
        return df, df