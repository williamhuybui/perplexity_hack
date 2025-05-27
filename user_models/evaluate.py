from openai import OpenAI
import re, random, json

API_KEY = "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"    
client = OpenAI(api_key=API_KEY, base_url="https://api.perplexity.ai")

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


def evaluate_answer(question, chunk, answer):
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
                """
            }
        ],
    )

    response_content = response.choices[0].message.content.strip()
    print("LLM Raw Output:", response_content)

    # Remove duplicate keys by keeping only the last occurrence
    cleaned_content = re.sub(
        r'(,\s*")(\w+_score)":\s*\d,\s*"\2":\s*\d',
        lambda m: f',{m.group(2)}": {m.group(0).split(":")[-1]}',
        response_content
    )

    result = json.loads(cleaned_content)
    return result