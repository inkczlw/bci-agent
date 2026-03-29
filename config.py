import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def get_llm():
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    return ChatOpenAI(
        model="deepseek-chat",
        api_key=str(api_key),
        base_url=str(base_url),
    )