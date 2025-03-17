from data_loader import load_and_split_documents
from embeddings import init_embeddings, create_vectorstore
from tools import get_web_search_tool
from graph import build_workflow

def main():
    documents = load_and_split_documents("./example/牙周病診治健康照護手冊.pdf")
    
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
    
    run("牙周病要手術治療的話需要花多少錢")

if __name__ == "__main__":
    main()