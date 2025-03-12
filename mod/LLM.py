import os
from groq import Groq

def get_response_from_LLM(prompt, api_key): #this function to be filled in with code that would return the response of the LLM for the input prompt
    groq_api_key = api_key

    client = Groq(
        # This is the default and can be omitted
        api_key=groq_api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",
    )

    return chat_completion.choices[0].message.content

def get_rationale_from_LLM(abstract, conference, api_key, n_words=100):
    prompt = f'Why is the research paper with abstract: {abstract} best suited to be published in {conference} conference in less than {n_words} words'
    rationale = get_response_from_LLM(prompt, api_key)
    return rationale