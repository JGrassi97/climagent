from langchain_openai import AzureChatOpenAI
import os

def initialize_llm():
    return AzureChatOpenAI(
        deployment_name=os.getenv("GPT_DEPLOYMENT_NAME"),
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        temperature=0,
    )