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
    documents: List[dict]
    web_search_count: int
    rag_retry_count: int
    next_step: str

def retrieve(state, retriever):
    print("---RETRIEVE---")
    try:
        results = retriever.invoke(state["question"])
        documents = []
        for result in results:
            documents.append({"text": result, "metadata": result.metadata if hasattr(result, "metadata") else {}})
        print(f"Retrieved {len(documents)} documents")
        return {
            "documents": documents,
            "question": state["question"],
            "web_search_count": state.get("web_search_count", 0),
            "rag_retry_count": state.get("rag_retry_count", 0)
        }
    except Exception as e:
        print(f"Error in retrieve: {str(e)}")
        raise

def web_search(state, web_search_tool):
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"] if state["documents"] else []
    web_search_count = state.get("web_search_count", 0) + 1
    
    if web_search_count > 2:
        print("---MAX SEARCH LIMIT REACHED, NO RELEVANT RESULTS---")
        return {
            "documents": documents,
            "question": question,
            "web_search_count": web_search_count,
            "rag_retry_count": state.get("rag_retry_count", 0)
        }
    
    enhanced_query = f"{question} Japanese translation site:*.edu site:*.org site:*.gov -inurl:(signup | login)"
    if "translation" in question.lower():
        enhanced_query += " BIOS terminology OR technical terms"
    docs = web_search_tool.invoke({"query": enhanced_query})
    web_results = [{"text": d["content"], "metadata": {}} for d in docs]
    return {
        "documents": documents + web_results,
        "question": question,
        "web_search_count": web_search_count,
        "rag_retry_count": state.get("rag_retry_count", 0)
    }

def grade_documents(state):
    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    web_search_count = state.get("web_search_count", 0)

    filtered_docs = []
    grader = get_document_grader()
    for d in documents:
        score = grader.invoke({"question": question, "document": d["text"]})
        grade = score.binary_score
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
    return {
        "documents": filtered_docs,
        "question": question,
        "web_search_count": web_search_count,
        "rag_retry_count": state.get("rag_retry_count", 0)
    }

def rag_generate(state):
    print("---GENERATE IN RAG MODE---")
    chain = get_rag_chain()
    doc_texts = [d["text"] for d in state["documents"]]
    generation = chain.invoke({"documents": doc_texts, "question": state["question"]})
    return {
        "documents": state["documents"],
        "question": state["question"],
        "generation": generation,
        "web_search_count": state.get("web_search_count", 0),
        "rag_retry_count": state.get("rag_retry_count", 0)
    }

def decide_rag_retry(state):
    print("---DECIDE RAG RETRY---")
    retry_count = state.get("rag_retry_count", 0) + 1
    max_retries = 3
    web_search_count = state.get("web_search_count", 0)

    if retry_count >= max_retries:
        if web_search_count < 2:  # 如果 Web Search 次數未達上限，切換到 Web Search
            print(f"---MAX RETRIES ({max_retries}) REACHED, SWITCHING TO WEB SEARCH---")
            return {
                "documents": state["documents"],
                "question": state["question"],
                "generation": state["generation"],
                "web_search_count": web_search_count,
                "rag_retry_count": 0,  # 重置計數
                "next_step": "web_search"
            }
        else:  # 如果 Web Search 次數已達上限，切換到 Plain Answer
            print(f"---MAX RETRIES ({max_retries}) REACHED AND MAX WEB SEARCH LIMIT REACHED, SWITCHING TO PLAIN ANSWER---")
            return {
                "documents": state["documents"],
                "question": state["question"],
                "generation": state["generation"],
                "web_search_count": web_search_count,
                "rag_retry_count": 0,  # 重置計數
                "next_step": "plain_answer"
            }
    else:
        print(f"---RETRYING RAG GENERATION ({retry_count}/{max_retries})---")
        return {
            "documents": state["documents"],
            "question": state["question"],
            "generation": state["generation"],
            "web_search_count": web_search_count,
            "rag_retry_count": retry_count,
            "next_step": "rag_generate"
        }

def plain_answer(state):
    print("---GENERATE PLAIN ANSWER---")
    chain = get_plain_chain()
    generation = chain.invoke({"question": state["question"]})
    return {
        "question": state["question"],
        "generation": generation,
        "web_search_count": state.get("web_search_count", 0),
        "rag_retry_count": 0
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
    doc_texts = [d["text"] for d in state["documents"]]
    hallucination_score = hallucination_grader.invoke({"documents": doc_texts, "generation": state["generation"]})

    if hallucination_score.binary_score == "no":
        answer_score = answer_grader.invoke({"question": state["question"], "generation": state["generation"]})
        if answer_score.binary_score == "yes":
            print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS AND USEFUL-")
            return "useful"
        else:
            print("  -DECISION: GENERATION IS GROUNDED IN DOCUMENTS BUT NOT USEFUL-")
            return "not useful"
    else:
        print("  -DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS-")
        return "not_grounded"

def decide_after_not_useful(state):
    web_search_count = state.get("web_search_count", 0)
    if web_search_count >= 2:
        print("---搜尋次數已達上限，切換到簡單回答---")
        return {"next_step": "plain_answer"}
    else:
        print("---切換到網路搜尋---")
        return {"next_step": "web_search"}

def decide_next_step(state):
    next_step = state.get("next_step")
    if next_step == "rag_generate":
        return "rag_generate"
    elif next_step == "web_search":
        return "web_search"
    elif next_step == "plain_answer":
        return "plain_answer"
    else:
        raise ValueError(f"Unknown next step: {next_step}")

def build_workflow(retriever, web_search_tool):
    workflow = StateGraph(GraphState)

    workflow.add_node("retrieve", lambda state: retrieve(state, retriever))
    workflow.add_node("web_search", lambda state: web_search(state, web_search_tool))
    workflow.add_node("plain_answer", plain_answer)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rag_generate", rag_generate)
    workflow.add_node("decide_rag_retry", decide_rag_retry)
    workflow.add_node("decide_after_not_useful", decide_after_not_useful)
    
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
        {
            "not_grounded": "decide_rag_retry",
            "not useful": "decide_after_not_useful", 
            "useful": END
        }
    )
    workflow.add_conditional_edges(
        "decide_rag_retry",
        decide_next_step,
        {
            "rag_generate": "rag_generate",
            "web_search": "web_search",
            "plain_answer": "plain_answer"
        }
    )
    workflow.add_conditional_edges(
        "decide_after_not_useful",
        lambda state: state.get("next_step"),
        {
            "web_search": "web_search",
            "plain_answer": "plain_answer"
        }
    )

    workflow.add_edge("plain_answer", END)
    
    return workflow.compile()