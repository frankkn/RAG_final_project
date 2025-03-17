from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from chains import (
    get_question_router, get_rag_chain, get_plain_chain,
    get_retrieval_grader, get_hallucination_grader, get_answer_grader
)

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]

def retrieve(state, retriever):
    print("---RETRIEVE---")
    documents = retriever.invoke(state["question"])
    return {"documents": documents, "question": state["question"]}

def web_search(state, web_search_tool):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] if state["documents"] else []
    if "web_search_count" not in state:
        state["web_search_count"] = 0
    state["web_search_count"] += 1
    if state["web_search_count"] > 2:  # 最多搜尋 2 次
        print("---MAX SEARCH LIMIT REACHED, NO RELEVANT RESULTS---")
        return {"documents": [], "question": question}
    docs = web_search_tool.invoke({"query": question})
    web_results = [Document(page_content=d["content"]) for d in docs]
    return {"documents": documents + web_results, "question": question}

def retrieval_grade(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    documents = state["documents"]
    question = state["question"]
    filtered_docs = []
    grader = get_retrieval_grader()
    for d in documents:
        score = grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            print("  -GRADE: DOCUMENT RELEVANT-")
            filtered_docs.append(d)
        else:
            print("  -GRADE: DOCUMENT NOT RELEVANT-")
    return {"documents": filtered_docs, "question": question}

def rag_generate(state):
    print("---GENERATE IN RAG MODE---")
    chain = get_rag_chain()
    generation = chain.invoke({"documents": state["documents"], "question": state["question"]})
    return {"documents": state["documents"], "question": state["question"], "generation": generation}

def plain_answer(state):
    print("---GENERATE PLAIN ANSWER---")
    chain = get_plain_chain()
    generation = chain.invoke({"question": state["question"]})
    return {"question": state["question"], "generation": generation}

def route_question(state):
    router = get_question_router()
    source = router.invoke({"question": state["question"]})
    if "tool_calls" not in source.additional_kwargs:
        return "plain_answer"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise ValueError("Router could not decide source")
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    return "web_search" if datasource == "web_search" else "vectorstore"

def route_retrieval(state):
    if not state["documents"]:
        print("---NO RELEVANT DOCUMENTS FOUND, SWITCHING TO PLAIN ANSWER---")
        return "plain_answer"  # 直接生成通用答案
    return "rag_generate"

def grade_rag_generation(state):
    """
    Returns:
        "useful": 答案基於文件內容且有效回答了問題，是一個理想的結果。
        "not useful": 答案基於文件內容，但未有效回答問題，可能需要重新生成或搜尋更多資料。
        "not supported": 答案是虛構的，不基於文件內容，無法信任。
    """
    hallucination_grader = get_hallucination_grader()
    answer_grader = get_answer_grader()
    hallucination_score = hallucination_grader.invoke({"documents": state["documents"], "generation": state["generation"]})
    if hallucination_score.binary_score == "no":
        answer_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        if answer_score.binary_score == "yes":
            print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS AND USEFUL-")
            return "useful" 
        else:
            print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS BUT NOT USEFUL-")
            return "not useful"
    print("  -DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS-")
    return "not supported"

def build_workflow(retriever, web_search_tool):
    workflow = StateGraph(GraphState)
    workflow.add_node("web_search", lambda state: web_search(state, web_search_tool))
    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("retrieval_grade", retrieval_grade)
    workflow.add_node("rag_generate", rag_generate)
    workflow.add_node("plain_answer", plain_answer)
    
    workflow.set_conditional_entry_point(
        route_question,
        {"web_search": "web_search", "vectorstore": "retrieve", "plain_answer": "plain_answer"}
    )
    workflow.add_edge("retrieve", "retrieval_grade")
    workflow.add_edge("web_search", "retrieval_grade")
    workflow.add_conditional_edges(
        "retrieval_grade",
        route_retrieval,
        {"web_search": "web_search", 
         "rag_generate": "rag_generate",
         "plain_answer": "plain_answer"}
    )
    workflow.add_conditional_edges(
        "rag_generate",
        grade_rag_generation,
        {"not supported": "rag_generate", "not useful": "web_search", "useful": END}
    )
    workflow.add_edge("plain_answer", END)
    
    return workflow.compile()