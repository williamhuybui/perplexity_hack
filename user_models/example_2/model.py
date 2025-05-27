import pandas as pd
import os, json, csv 

LOG_FILE = "qa_log_2.csv"

def process_answer(qa_chain, query):
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
    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=["question", "answer", "sources", "top_k_chunks"])
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "question": query,
            "answer": answer_text,
            "sources": "; ".join(list(set(sources_list))),
            "top_k_chunks": "; ".join(list(set(top_k_chunks)))
        })


def qa(qa_chain):
    questions= pd.read_csv("session_1/questions.csv")

    for index, row in questions.iterrows():
        question = row['question']
        process_answer(qa_chain, question)

    df = pd.read_csv(LOG_FILE)
    return df

def finalize(evaluate_answer, df):
    final_rows = []

    for _, row in df.iterrows():
        question = row['question']
        top_k_chunk = row['top_k_chunks']
        answer = row['answer']

        success = False
        while not success:
            try:
                evaluation = evaluate_answer(question, top_k_chunk, answer)
                success = True  # Break loop if successful
            except Exception as e:
                print(f"Retrying for question: {question} due to error: {e}")

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
    final_df.to_csv('final.csv', index=False)
    print("Saved final results to final.csv")
    return final_df