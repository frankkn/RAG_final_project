from typing import List
from typing_extensions import TypedDict
from langchain.schema import Document
from langgraph.graph import END, StateGraph
from chains import (
    get_question_router, get_rag_chain, get_plain_chain,
    get_document_grader, get_hallucination_grader, get_answer_grader
)

class GraphState(TypedDict):
    question: str
    generation: str
    documents: List[str]
    web_search_count: int

def retrieve(state, retriever):
    print("---RETRIEVE---")
    try:
        documents = retriever.invoke(state["question"])
        print(f"Retrieved {len(documents)} documents")
        return {
            "documents": documents,
            "question": state["question"],
            "web_search_count": state.get("web_search_count", 0)
        }
    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        raise

def web_search(state, web_search_tool):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] if state["documents"] else []
    web_search_count = state.get("web_search_count", 0) + 1
    
    # print(f"  - Current web_search_count: {web_search_count}")
    if web_search_count > 2:
        print("---MAX SEARCH LIMIT REACHED, NO RELEVANT RESULTS---")
        return {
            "documents": documents,
            "question": question,
            "web_search_count": web_search_count
        }
    
    docs = web_search_tool.invoke({"query": question})
    web_results = [Document(page_content=d["content"]) for d in docs]
    return {
        "documents": documents + web_results,
        "question": question,
        "web_search_count": web_search_count
    }

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    web_search_count = state.get("web_search_count", 0)
    # print(f"  - Web search count in grade_documents: {web_search_count}")

    filtered_docs = []
    grader = get_document_grader()
    for d in documents:
        score = grader.invoke({"question": question, "document": d.page_content})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_count": web_search_count
    }

def rag_generate(state):
    print("---GENERATE IN RAG MODE---")
    chain = get_rag_chain()
    generation = chain.invoke({"documents": state["documents"], "question": state["question"]})
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": generation,
        "web_search_count": state.get("web_search_count", 0)
    }

def plain_answer(state):
    print("---GENERATE PLAIN ANSWER---")
    chain = get_plain_chain()
    generation = chain.invoke({"question": state["question"]})
    return {
        "question": state["question"],
        "generation": generation,
        "web_search_count": state.get("web_search_count", 0)
    }

def route_question(state):
    router = get_question_router()
    source = router.invoke({"question": state["question"]})
    if "tool_calls" not in source.additional_kwargs:
        print("---ROUTED TO PLAIN ANSWER---")
        return "plain_answer"
    if len(source.additional_kwargs["tool_calls"]) == 0:
        raise ValueError("Router could not decide source")
    datasource = source.additional_kwargs["tool_calls"][0]["function"]["name"]
    if datasource == "web_search":
        print("---ROUTED TO WEB SEARCH---")
        return "web_search" 
    else:
        print("---ROUTED TO VECTORSTORE---")
        return "vectorstore"

def decide_to_generate(state):
    web_search_count = state.get("web_search_count", 0)
    # print(f"---DECIDE TO GENERATE, Web search count: {web_search_count}---")
    
    if not state["documents"]:
        print("---NO RELEVANT DOCUMENTS FOUND---")
        if web_search_count < 2:
            print("---SWITCHING TO WEB SEARCH---")
            return "web_search"
        else:
            print("---MAX SEARCH LIMIT REACHED, SWITCHING TO PLAIN ANSWER---")
            return "plain_answer"
    print("---FOUND RELEVANT DOCUMENTS, PROCEEDING TO RAG GENERATION---")
    return "rag_generate"

def grade_rag_generation(state):
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

    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("web_search", lambda state: web_search(state, web_search_tool))
    workflow.add_node("plain_answer", plain_answer)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rag_generate", rag_generate)
    
    workflow.set_conditional_entry_point(
        route_question,
        {"web_search": "web_search",
         "vectorstore": "retrieve",
         "plain_answer": "plain_answer"}
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_edge("web_search", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
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