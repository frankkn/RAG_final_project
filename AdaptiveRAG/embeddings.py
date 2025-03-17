from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import get_model_config

def init_embeddings(model_version="text-embedding-ada-002"):
    config = get_model_config(model_version)
    return AzureOpenAIEmbeddings(
        openai_api_key=config['api_key'],
        azure_endpoint=config['api_base'],
        openai_api_type=config['openai_type'],
        openai_api_version=config['api_version'],
        azure_deployment=config['deployment_name']
    )

def create_vectorstore(documents, embeddings):
    vectorstore = Chroma.from_documents(documents=documents, embedding=embeddings)
    return vectorstore.as_retriever()