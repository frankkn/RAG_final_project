from data_loader import load_and_split_documents
from embeddings import init_embeddings, get_or_create_vectorstore
from tools import get_web_search_tool
from graph import build_workflow
import os

def main():
    embeddings = init_embeddings()

    file_path = "./example/2025_ML_UNI_20250311.xlsx"
    persist_directory = "./chroma_db"

    # 如果檔案存在，第一次執行時載入並建立資料庫；後續直接使用已有資料庫
    if os.path.exists(file_path):
        documents = load_and_split_documents(file_path) if not os.path.exists(persist_directory) else None
        retriever = get_or_create_vectorstore(embeddings, documents, persist_directory)
    else:
        raise FileNotFoundError(f"檔案 {file_path} 不存在，請確認路徑是否正確。")
    
    web_search_tool = get_web_search_tool()
    
    app = build_workflow(retriever, web_search_tool)

    def run(question):
        inputs = {"question": question}
        for output in app.stream(inputs):
            print("\n")
        if 'rag_generate' in output:
            print(output['rag_generate']['generation'])
        elif 'plain_answer' in output:
            print(output['plain_answer']['generation'])

    #1:如何設定Asset tag?它跟Service tag有區別嗎?      
    #2:請問哪些production line支持Boot indicator?
    #3:如何治療PTSD?
    while True:
        question = input("請輸入你的問題（輸入 'exit' 離開）：")

        if question.lower() == 'exit':
            print("Bye!")
            break

        run(question)

if __name__ == "__main__":
    main()