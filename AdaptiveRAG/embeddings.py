from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from config import get_model_config
import os
import time
from typing import List
from langchain.schema import Document

def init_embeddings(model_version="text-embedding-ada-002"):
    config = get_model_config(model_version)
    return AzureOpenAIEmbeddings(
        openai_api_key=config['api_key'],
        azure_endpoint=config['api_base'],
        openai_api_type=config['openai_type'],
        openai_api_version=config['api_version'],
        azure_deployment=config['deployment_name']
    )

def batch_documents(documents: List[Document], batch_size: int) -> List[List[Document]]:
    """將文件分批處理"""
    return [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]

def get_or_create_vectorstore(embeddings, documents=None, persist_directory="./chroma_db", batch_size=50):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
    
    # 如果提供了 documents 且資料庫不存在，則建立新資料庫
    if documents and not os.path.exists(os.path.join(persist_directory, "chroma.sqlite3")):
        print(f"Creating new vectorstore with {len(documents)} documents")
        batches = batch_documents(documents, batch_size)

        vectorstore = Chroma.from_documents(
            documents=batches[0],  # 只用第一批初始化
            embedding=embeddings,
            persist_directory=persist_directory
        )

        for i, batch in enumerate(batches[1:], start=1):
            print(f"Processing batch {i+1}/{len(batches)} with {len(batch)} documents")
            vectorstore.add_documents(documents=batch)

    else:
        # 否則載入現有資料庫
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    return vectorstore.as_retriever()