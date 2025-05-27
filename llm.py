from openai import OpenAI

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
    messages= [{
        "role": "system",
                "content": """
                You are a financial report assistant. 
                I will provide the first few pages of a financial report, and your task is to give a concise, single-sentence summary answering: 
                (1) which company the report is about and 
                (2) what year it covers. 
                Limit the summary to 50 words, with no extra details or formatting.
                """
    },
        {   
            "role": "user",
            "content":  f"""
            The first few pages {first_n_pages}
            Your response: 
            """
            
        },
    ])
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
        messages=[{
            "role": "system",
            "content": """
            You are a question generator. 
            I will provide a chunk of information along with its PDF context. 
            Your task is to generate one question  with the following requirement
            (1) The question should based solely on the chunks content
            (2) The question should include enough context from the summary (company name and year) to make it clear what the question is about.
            (3) Do not add any extra information. 
            (4) If the chunk lacks useful content, respond with an empty string.
            """
        },
        {
            "role": "user",
            "content": f"""
            PDF Summary {summary}. Chunk Text: {chunk}
            Your question:
            """
        }],
    )
    return response.choices[0].message.content.strip()