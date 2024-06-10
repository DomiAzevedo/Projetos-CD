import openai

class LLM_Model:
    def __init__(self, api_key):
        openai.api_key = api_key

    def generate_generic_questions(self, book_title, book_description):
        prompt = f"Generate three generic questions about any book, given the following book titled '{book_title}' with this description: '{book_description}'. The questions should be broad and not specific, not giving away the book title, being more general and applicable to other books, but at the same time giving elements to discuss the book."
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=150,
            temperature=0.7,
        )
        questions = response.choices[0].message['content'].strip().split("\n")
        return [q for q in questions if q]
