import os

from dotenv import load_dotenv
from autogen_ext.models.openai import OpenAIChatCompletionClient

# Ensure environment variables are loaded
load_dotenv()
def get_model_client():
    api_key = os.getenv("MISTRAL_API_KEY")
    base_url = os.getenv("MISTRAL_BASE_URL", "https://api.mistral.ai/v1")

    return OpenAIChatCompletionClient(
        model="mistral-medium-latest",
        api_key=api_key,
        base_url=base_url,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "mistral",
            "structured_output": False# Set to False to avoid role-order conflicts during tool parsing
        }
    )

# def get_model_client():
#     return OpenAIChatCompletionClient(
#         model="llama3.1:latest", # Updated from qwen2.5:14b
#         timeout=600,
#         api_key="ollama",
#         base_url="http://localhost:11434/v1",
#         model_info={
#             "vision": False,
#             "function_calling": True,
#             "json_output": True,
#             "family": "llama3", # Updated family
#             "structured_output": False,
#         }
#     )
