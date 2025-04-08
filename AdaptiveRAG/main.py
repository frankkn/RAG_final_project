from data_loader import load_excel_data, save_excel_data
from embeddings import init_embeddings, get_or_create_vectorstore
from tools import get_web_search_tool
from graph import build_workflow
from translator import translate_missing_fields
import os

def main():
    embeddings = init_embeddings()

    file_path = "./example/2025_ML_UNI_20250311.xlsx"
    persist_directory = "./chroma_db"
    output_file_path = "./example/2025_ML_UNI_20250311_translated.xlsx"

    if os.path.exists(file_path):
        data_list = load_excel_data(file_path)
        retriever = get_or_create_vectorstore(embeddings, data_list, persist_directory)
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

    def translate_lost_string():
        print("Translating missing fields in Lost_String sheet...")
        updated_data_list = translate_missing_fields(data_list)
        save_excel_data(file_path, updated_data_list, output_file_path)
        print(f"Translation completed. Updated file saved to: {output_file_path}")

    while True:
        print("\n請選擇操作：")
        print("1. 提問問題（輸入問題）")
        print("2. 自動翻譯 Lost_String 分頁")
        print("3. 離開（輸入 'exit'）")
        choice = input("輸入你的選擇（1/2/3）：")

        if choice == '1':
            question = input("請輸入你的問題：")
            run(question)
        elif choice == '2':
            translate_lost_string()
        elif choice == '3' or choice.lower() == 'exit':
            print("Bye!")
            break
        else:
            print("無效的選擇，請重新輸入。")

if __name__ == "__main__":
    main()