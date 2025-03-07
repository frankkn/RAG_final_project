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

def init_prompt_parser():
    str_parser = StrOutputParser()
    template = (
        "請根據以下內容加上自身判斷回答問題:\n"
        "{context}\n"
        "問題: {question}"
    )
    prompt = ChatPromptTemplate.from_template(template)
    
    return prompt, str_parser

def load_file(file_path):
    loader = UnstructuredFileLoader(file_path)
    doc = loader.load()
    return doc

def generate_db(file_path, db_path, collection_name):
    gpt_embed_version = 'text-embedding-ada-002'
    gpt_emb_config = get_model_configuration(gpt_embed_version)

    chroma_client = chromadb.PersistentClient(path=db_path)
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

    if collection == 0:
        doc = load_file(file_path)

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n\n"],
            chunk_size=150,
            chunk_overlap=0
        )
        splits = text_splitter.split_documents(doc)

        for i in splits[:3]:
            print(i.page_content)
            print('_'*10)
    
        texts = [split.page_content for split in splits]
        ids = [str(uuid.uuid4()) for _ in splits]
        collection.add(documents=texts, ids=ids)
    
    return collection

def rag(collection, chat_model, prompt, str_parser):
    def retrieve(question):
        results = collection.query(query_texts=[question], n_results=5)
        documents = results.get("documents", [[]])[0]  # 取出第一組結果的文本
        return "\n".join(documents)
    
    chain = (
        {"context": retrieve, "question": RunnablePassthrough()}
        | prompt
        | chat_model
        | str_parser
    )
    return chain

if __name__ == "__main__":
    chat_model= init_model()
    prompt, str_parser = init_prompt_parser()

    db_path = "./"
    file_path = "example_excel.xlsx"
    # doc = load_file(file_path)
    # pprint(doc)
    collection_name = "AMUSEMENT"
    collection = generate_db(file_path, db_path, collection_name)
    
    excel_chain = rag(collection, chat_model, prompt, str_parser)
    question = "我身高超過 120 公分, 我可玩那些遊樂設施?"
    print(excel_chain.invoke(question))