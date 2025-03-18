from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from config import get_model_config
import os

def init_embeddings(model_version="text-embedding-ada-002"):
    config = get_model_config(model_version)
    return AzureOpenAIEmbeddings(
        openai_api_key=config['api_key'],
        azure_endpoint=config['api_base'],
        openai_api_type=config['openai_type'],
        openai_api_version=config['api_version'],
        azure_deployment=config['deployment_name']
    )

def get_or_create_vectorstore(embeddings, documents=None, persist_directory="./chroma_db"):
    # 確保持久化目錄存在
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # 如果提供了 documents 且資料庫不存在，則建立新資料庫
    if documents and not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    else:
        # 否則載入現有資料庫
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    return vectorstore.as_retriever()