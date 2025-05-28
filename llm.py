from openai import OpenAI
import re, json


def llm_summary_extraction(api_key, first_n_pages):
    """
    Extracts a summary from the first few pages of a financial report.
    Args:
        api_key (str): API key for the OpenAI client.
        first_n_pages (str): The first few pages of the financial report text.
    Returns:
        str: A concise summary of the report, including the company name and year.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")

    response = client.chat.completions.create(
        model="sonar",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": """
                You are a financial report assistant. 
                I will provide the first few pages of a financial report, and your task is to give a concise, single-sentence summary answering: 
                (1) which company the report is about and 
                (2) what year it covers. 
                Limit the summary to 50 words, with no extra details or formatting.
                """,
            },
            {
                "role": "user",
                "content": f"""
            The first few pages {first_n_pages}
            Your response: 
            """,
            },
        ],
    )
    return response.choices[0].message.content.strip()


def llm_generate_questions(api_key, summary, chunk):
    """
    Generates a question based on the provided summary and chunk of text.
    Args:
        api_key (str): API key for the OpenAI client.
        summary (str): A concise summary of the financial report.
        chunk (str): A chunk of text from the financial report.
    """
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {
                "role": "system",
                "content": """
            You are a question generator. 
            I will provide a chunk of information along with its PDF context. 
            Your task is to generate one question  with the following requirement
            (1) The question should based solely on the chunks content
            (2) The question should include enough context from the summary (company name and year) to make it clear what the question is about.
            (3) Do not add any extra information. 
            (4) If the chunk lacks useful content, respond with an empty string.
            """,
            },
            {
                "role": "user",
                "content": f"""
            PDF Summary {summary}. Chunk Text: {chunk}
            Your question:
            """,
            },
        ],
    )
    return response.choices[0].message.content.strip()


def llm_evaluate_answer(api_key, question, chunk, answer):
    client = OpenAI(api_key=api_key, base_url="https://api.perplexity.ai")
    prompt = """
    You are a financial data Q&A evaluator.

    You are given:
    - A **question** generated from a document chunk.
    - The **document chunk** (ground truth source).
    - A **model-generated answer** to the question.

    Your job is to score the model’s answer by carefully comparing it to the document chunk.

    Use the following rubric for each category:

    ---
    **Factual Correctness**
    - 5 = All facts are fully correct and consistent with the chunk.
    - 4 = Minor factual inaccuracies but mostly correct.
    - 3 = Some factual inaccuracies, partly correct.
    - 2 = Major factual mistakes, mostly incorrect.
    - 1 = Completely factually wrong.

    ---
    **Completeness**
    - 5 = Fully answers the question with all key details.
    - 4 = Mostly complete, missing minor details.
    - 3 = Partially complete, missing important parts.
    - 2 = Mostly incomplete, only touches on part of the question.
    - 1 = Completely incomplete.

    ---
    3**Clarity**
    - 5 = Clear, precise, and easy to understand.
    - 4 = Mostly clear, with minor awkwardness.
    - 3 = Understandable but somewhat confusing or vague.
    - 2 = Hard to understand or poorly phrased.
    - 1 = Completely unclear or nonsensical.

    ---
    **Response Format**
    Return ONLY this JSON (no extra explanation):
    {
        "factual_correctness_score": [1-5],
        "completeness_score": [1-5],
        "clarity_score": [1-5],
        "comments": "A brief explanation (1-2 sentences) why you assigned these scores."
    }
"""
    response = client.chat.completions.create(
        model="sonar",
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"""
                Please evaluate the following answer based on the provided question and document chunk. 
                Return ONLY a valid JSON object.

                Question: {question}

                Document Chunk: {chunk}

                Model Answer: {answer}
                """,
            },
        ],
    )

    response_content = response.choices[0].message.content.strip()

    # Remove duplicate keys by keeping only the last occurrence
    cleaned_content = re.sub(
        r'(,\s*")(\w+_score)":\s*\d,\s*"\2":\s*\d',
        lambda m: f',{m.group(2)}": {m.group(0).split(":")[-1]}',
        response_content,
    )

    result = json.loads(cleaned_content)
    return result
