from data_loader import load_and_split_documents
from embeddings import init_embeddings, create_vectorstore
from tools import get_web_search_tool
from graph import build_workflow

def main():
    documents = load_and_split_documents("./example/Y2024H2 Intel Platform_Commercial_BIOS_Setup_Menu_Specification_V2.0.7.pdf")
    
    embeddings = init_embeddings()
    retriever = create_vectorstore(documents, embeddings)
    
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

    #1:什麼是secure boot?如何開啟它?      
    #2:什麼是WAKE ON LAN?
    #3:如何治療PTSD?
    while True:
        question = input("請輸入你的問題（輸入 'exit' 離開）：")

        if question.lower() == 'exit':
            print("Bye!")
            break

        run(question)

if __name__ == "__main__":
    main()