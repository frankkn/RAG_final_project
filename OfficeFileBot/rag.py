from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import AzureChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_community.document_loaders import PyPDFLoader
import pandas as pd
import os
import chromadb
import uuid
import hashlib
from config import get_model_configuration
from loaders import FileLoader
from taipy.gui import notify

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

def splitter(docs, separators, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap)
    splits = text_splitter.split_documents(docs)
    return splits

def get_file_hash(file_path):
    with open(file_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def generate_db(db_path, collection_name, splits, file_path, state=None):
    gpt_embed_version = 'text-embedding-ada-002'
    gpt_emb_config = get_model_configuration(gpt_embed_version)
    chroma_client = chromadb.PersistentClient(path=db_path)

    current_hash = get_file_hash(file_path)

    try:
        collection = chroma_client.get_collection(name=collection_name)
        stored_hash = collection.metadata.get("file_hash") if collection.metadata else None
        if stored_hash == current_hash:
            print(f"Reusing existing collection: {collection_name}")
            if state is not None:
                notify(state, "info", "此檔案已存在於資料庫，將重用現有向量資料庫。")
            return collection, True
        else:
            print(f"File content changed, regenerating collection: {collection_name}")
            chroma_client.delete_collection(name=collection_name)
    except Exception:
        print(f"No existing collection found, creating new one: {collection_name}")

    from chromadb.utils import embedding_functions
    try:
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=gpt_emb_config['api_key'],
            api_base=gpt_emb_config['api_base'],
            api_type=gpt_emb_config['openai_type'],
            api_version=gpt_emb_config['api_version'],
            deployment_id=gpt_emb_config['deployment_name']
        )
        print("OpenAI embedding function initialized successfully with dimension 1536")
    except Exception as e:
        print(f"Failed to initialize OpenAI embedding function: {str(e)}")
        raise ValueError("Azure OpenAI embedding initialization failed. Check .env configuration.")

    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "file_hash": current_hash},
        embedding_function=openai_ef
    )
    print(f"Collection created with embedding function: {openai_ef.__class__.__name__}")
    
    texts = [split.page_content for split in splits]
    ids = [str(uuid.uuid4()) for _ in splits]
    batch_size = 10
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        collection.add(documents=batch_texts, ids=batch_ids)
        if state is not None:
            notify(state, "info", f"處理批量 {i // batch_size + 1}/{len(texts) // batch_size + 1}")
    return collection, False

def rag(splits, collection_name, file_path, state=None):
    collection, reused = generate_db("./", collection_name, splits, file_path, state)
    chat_model = init_model()
    prompt, str_parser = init_prompt_parser()
    
    def retrieve(question):
        results = collection.query(query_texts=[question], n_results=5)
        documents = results.get("documents", [[]])[0]
        return "\n".join(documents)
    
    chain = (
        {"context": retrieve, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | str_parser
    )
    return chain, reused

def RAG(state):
    notify(state, "info", f"開始處理檔案: {os.path.basename(state.content)}")
    try:
        docs = FileLoader.load(state.content)
        notify(state, "success", f"檔案載入完成，共 {len(docs)} 頁/文件")
    except Exception as e:
        notify(state, "error", f"檔案載入失敗: {str(e)}")
        return

    notify(state, "info", "正在分割段落...")
    try:
        splits = splitter(docs, eval(f"[{state.separators}]"), int(state.chunk_size), int(state.chunk_overlap))
        notify(state, "success", f"分割完成，生成了 {len(splits)} 個片段")
    except Exception as e:
        notify(state, "error", f"分割段落失敗: {str(e)}")
        return

    notify(state, "info", "正在檢查檔案是否已存在於資料庫...")
    try:
        collection_name = os.path.splitext(os.path.basename(state.content))[0]
        state.chain, reused = rag(splits, collection_name, state.content, state)
        if not reused:
            notify(state, "success", "向量資料庫生成完成，可以開始提問！")
    except Exception as e:
        notify(state, "error", f"轉向量失敗: {str(e)}")

def csv_file(state):
    if 'csv' in state.content and state.skiprows is not None:
        state.is_csv = True
        state.chain = pandas_agent(state.content, int(state.skiprows))
    elif not 'csv' in state.content:
        state.is_csv = False
    else:
        notify(state, "error", "請先輸入 CSV 檔案參數")

def pandas_agent(path, skiprows):
    df = pd.read_csv(path, skiprows=skiprows)
    agent = create_pandas_dataframe_agent(llm=chat_model,
                                          df=df,
                                          prefix='回答請使用繁體中文',
                                          agent_type="openai-tools")
    return agent

def request(state, prompt):
    if state.chain:
        response = state.chain.invoke(prompt)
        if state.is_csv:
            return response['output']
        else:
            return response.replace("\n", "")
    else:
        notify(state, "error", "請先上傳檔案")
        return "請先上傳檔案"