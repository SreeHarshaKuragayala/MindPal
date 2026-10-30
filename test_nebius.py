
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("NEBIUS_BASE_URL"),
    api_key=os.getenv("NEBIUS_API_KEY")
)

response = client.chat.completions.create(
    model=os.getenv("NEBIUS_MODEL"),
    messages=[
        {
            "role": "system",
            "content": "You are MindPal, a supportive wellness companion."
        },
        {
            "role": "user",
            "content": "Hello MindPal! Are you connected?"
        }
    ]
)

print(response.choices[0].message.content)
