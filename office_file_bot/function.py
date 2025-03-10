from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import os

import chromadb
import uuid
from rich import print as pprint
from model_configurations import get_model_configuration
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_unstructured import UnstructuredLoader
from langchain_community.document_loaders import UnstructuredFileLoader
from chromadb.utils import embedding_functions

def init_model():
    gpt_chat_version = 'gpt-4o'
    gpt_config = get_model_configuration(gpt_chat_version)

    chat_model = AzureChatOpenAI(
        model=gpt_config['model_name'],
        deployment_name=gpt_config['deployment_name'],
        openai_api_key=gpt_config['api_key'],
        openai_api_version=gpt_config['api_version'],
        azure_endpoint=gpt_config['api_base'],
        temperature=gpt_config['temperature']
    )

    return chat_model

# embeddings = OpenAIEmbeddings()
chat_model = init_model()

def init_prompt_parser():
    str_parser = StrOutputParser()
    template = (
        "請根據以下內容加上自身判斷回答問題:\n"
        "{context}\n"
        "問題: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt, str_parser

prompt, str_parser = init_prompt_parser()

def office_file(file_path):
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    return docs

def splitter(docs, separators, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits

def generate_db(db_path, collection_name, splits):
    print("Initializing ChromaDB client...")
    gpt_embed_version = 'text-embedding-ada-002'
    try:
        gpt_emb_config = get_model_configuration(gpt_embed_version)
        chroma_client = chromadb.PersistentClient(path=db_path)
        print("ChromaDB client initialized")

        print("Setting up OpenAI embedding function...")
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=gpt_emb_config['api_key'],
            api_base=gpt_emb_config['api_base'],
            api_type=gpt_emb_config['openai_type'],
            api_version=gpt_emb_config['api_version'],
            deployment_id=gpt_emb_config['deployment_name']
        )
        print("OpenAI embedding function set up")

        print("Creating or getting collection...")
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=openai_ef
        )
        print("Collection ready")

        # if collection.count() == 0:
        print("Preparing documents for embedding...")
        texts = [split.page_content for split in splits]
        ids = [str(uuid.uuid4()) for _ in splits]
        batch_size = 10
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_num = i // batch_size + 1
            try:
                collection.add(documents=batch_texts, ids=batch_ids)
            except Exception as batch_error:
                print(f"Error in batch {batch_num}: {str(batch_error)}")
                raise
        
        return collection
    
    except Exception as e:
        print(f"Database generation failed: {str(e)}")
        raise

def pdf_load(file_path):
    loader = PyPDFLoader(file_path=file_path,
                        extract_images=True)
    docs = loader.load()
    return docs

def rag(splits, collection_name):
    try:
        collection = generate_db("./", collection_name, splits)
        chat_model = init_model()
        prompt, str_parser = init_prompt_parser()
        
        def retrieve(question):
            print("Retrieving documents...")
            results = collection.query(query_texts=[question], n_results=5)
            documents = results.get("documents", [[]])[0]
            print(f"Retrieved {len(documents)} documents")
            return "\n".join(documents)
        
        print("Building RAG chain...")
        chain = (
            {"context": retrieve, "question": RunnablePassthrough()}
            | prompt
            | chat_model
            | str_parser
        )
        print("RAG chain built successfully")
        return chain
    except Exception as e:
        print(f"RAG process failed: {str(e)}")
        raise

def pandas_agent(path, skiprows):
    df = pd.read_csv(path,skiprows=skiprows)
    agent = create_pandas_dataframe_agent(llm=chat_model,
                                            df=df,
                                            prefix='回答請使用繁體中文',
                                            agent_type="openai-tools")
    return agent