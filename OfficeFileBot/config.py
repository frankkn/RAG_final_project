import os
from dotenv import load_dotenv
load_dotenv()

configurations = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_base": os.getenv('AZURE_OPENAI_GPT4O_ENDPOINT'),
        "api_key": os.getenv('AZURE_OPENAI_GPT4O_KEY'),
        "deployment_name": os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT_CHAT'),
        "api_version": os.getenv('AZURE_OPENAI_GPT4O_VERSION'),
        "temperature": 0.0,
        "top_p": 1.0,
        "max_token": 4096
    },
    "text-embedding-ada-002": {
        "api_base": os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
        "api_key": os.getenv('AZURE_OPENAI_EMBEDDING_KEY'),
        "deployment_name": os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDING'),
        "api_version": os.getenv('AZURE_OPENAI_VERSION'),
        "model": os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDING_MODEL_NAME'),
        "openai_type": os.getenv('AZURE_OPENAI_TYPE'),
    }
}

def get_model_configuration(model_version):
    return configurations.get(model_version)