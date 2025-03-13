from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
import os
import chromadb
import uuid
import hashlib
import math
from dotenv import load_dotenv
from taipy.gui import notify
from loaders import FileLoader

load_dotenv()

configurations = {
    "gpt-4o": {
        "model_name": "gpt-4o",
        "api_base": os.getenv('AZURE_OPENAI_GPT4O_ENDPOINT'),
        "api_key": os.getenv('AZURE_OPENAI_GPT4O_KEY'),
        "deployment_name": os.getenv('AZURE_OPENAI_GPT4O_DEPLOYMENT_CHAT'),
        "api_version": os.getenv('AZURE_OPENAI_GPT4O_VERSION'),
        "temperature": 0.0,
    },
    "text-embedding-ada-002": {
        "api_base": os.getenv('AZURE_OPENAI_EMBEDDING_ENDPOINT'),
        "api_key": os.getenv('AZURE_OPENAI_EMBEDDING_KEY'),
        "deployment_name": os.getenv('AZURE_OPENAI_DEPLOYMENT_EMBEDDING'),
        "api_version": os.getenv('AZURE_OPENAI_VERSION'),
        "openai_type": os.getenv('AZURE_OPENAI_TYPE'),
    }
}

def get_model_configuration(model_version):
    return configurations.get(model_version)

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
    try:
        collection = chroma_client.get_collection(name=collection_name)
        stored_hash_path = os.path.join(db_path, f"{collection_name}_hash.txt")
        if os.path.exists(stored_hash_path):
            with open(stored_hash_path, "r") as f:
                stored_hash = f.read().strip()
            current_hash = get_file_hash(file_path)
            if stored_hash == current_hash:
                print(f"Reusing existing collection: {collection_name}")
                return collection
            else:
                print(f"File content changed, regenerating collection: {collection_name}")
                chroma_client.delete_collection(name=collection_name)
    except Exception:
        print(f"No existing collection found, creating new one: {collection_name}")

    from chromadb.utils import embedding_functions
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=gpt_emb_config['api_key'],
        api_base=gpt_emb_config['api_base'],
        api_type=gpt_emb_config['openai_type'],
        api_version=gpt_emb_config['api_version'],
        deployment_id=gpt_emb_config['deployment_name']
    )
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )
    texts = [split.page_content for split in splits]
    ids = [str(uuid.uuid4()) for _ in splits]
    batch_size = min(50, max(10, len(texts) // 100))
    total_batches = math.ceil(len(texts) / batch_size)
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        if batch_texts:
            collection.add(documents=batch_texts, ids=batch_ids)
            if state is not None:
                current_batch = (i // batch_size) + 1
                notify(state, "info", f"處理批量 {current_batch}/{total_batches}")
    return collection, False

def rag(splits, collection_name, file_path):
    collection = generate_db("./", collection_name, splits, file_path)
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
    return chain

def pandas_agent(path, skiprows):
    df = pd.read_csv(path, skiprows=skiprows)
    agent = create_pandas_dataframe_agent(llm=chat_model,
                                          df=df,
                                          prefix='回答請使用繁體中文',
                                          agent_type="openai-tools",
                                          allow_dangerous_code=True)
    return agent

def RAG(state):
    notify(state, "info", f"開始處理檔案: {os.path.basename(state.content)}")
    try:
        is_csv, docs = FileLoader.load(state.content)
        if is_csv:
            notify(state, "info", "已識別為 CSV 檔案，將使用專用處理流程。")
            csv_file(state)
            return
        notify(state, "success", f"檔案載入完成，共 {len(docs)} 頁/文件")
        print(f'File loaded with {len(docs)} documents/pages')
    except Exception as e:
        notify(state, "error", f"檔案載入失敗: {str(e)}")
        return
    notify(state, "info", "正在分割段落...")
    try:
        splits = splitter(docs, eval(f"[{state.separators}]"), int(state.chunk_size), int(state.chunk_overlap))
        notify(state, "success", f"分割完成，生成了 {len(splits)} 個片段")
        print(f"Created {len(splits)} splits")
    except Exception as e:
        notify(state, "error", f"分割段落失敗: {str(e)}")
        return
    notify(state, "info", "正在生成向量資料庫...")
    try:
        collection_name = os.path.splitext(os.path.basename(state.content))[0]
        state.chain = rag(splits, collection_name, state.content)
        notify(state, "success", "向量資料庫生成完成，可以開始提問！")
        print('完成')
    except Exception as e:
        notify(state, "error", f"轉向量失敗: {str(e)}")
        print(f"錯誤: {str(e)}")

def csv_file(state):
    if 'csv' in state.content.lower() and state.skiprows is not None:
        state.is_csv = True
        state.chain = pandas_agent(state.content, int(state.skiprows))
        notify(state, "success", "CSV 檔案處理完成，可以開始提問！")
    elif 'csv' not in state.content.lower():
        state.is_csv = False
        notify(state, "info", "請點擊 'RAG 處理' 按鈕以生成向量資料庫。")
    else:
        notify(state, "error", "請先輸入 CSV 檔案參數")