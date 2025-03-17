from pydantic import BaseModel, Field
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from config import get_model_config
from prompts import (
    get_route_prompt, get_rag_prompt, get_plain_prompt,
    get_retrieval_grade_prompt, get_hallucination_grade_prompt, get_answer_grade_prompt
)

def init_llm(model_version="gpt-4o"):
    config = get_model_config(model_version)
    return AzureChatOpenAI(
        model=config['model_name'],
        deployment_name=config['deployment_name'],
        openai_api_key=config['api_key'],
        openai_api_version=config['api_version'],
        azure_endpoint=config['api_base'],
        temperature=config['temperature']
    )

class WebSearch(BaseModel):
    query: str = Field(description="使用網路搜尋時輸入的問題")

class Vectorstore(BaseModel):
    query: str = Field(description="搜尋向量資料庫時輸入的問題")

class GradeDocuments(BaseModel):
    binary_score: str = Field(description="請問文章與問題是否相關。('yes' or 'no')")

class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="答案是否由為虛構。('yes' or 'no')")

class GradeAnswer(BaseModel):
    binary_score: str = Field(description="答案是否回應問題。('yes' or 'no')")

def get_question_router():
    llm = init_llm()
    structured_llm_router = llm.bind_tools(tools=[WebSearch, Vectorstore])
    return get_route_prompt() | structured_llm_router

def get_rag_chain():
    llm = init_llm()
    return get_rag_prompt() | llm | StrOutputParser()

def get_plain_chain():
    llm = init_llm()
    return get_plain_prompt() | llm | StrOutputParser()

def get_retrieval_grader():
    llm = init_llm()
    structured_llm_grader = llm.with_structured_output(GradeDocuments)
    return get_retrieval_grade_prompt() | structured_llm_grader

def get_hallucination_grader():
    llm = init_llm()
    structured_llm_grader = llm.with_structured_output(GradeHallucinations)
    return get_hallucination_grade_prompt() | structured_llm_grader

def get_answer_grader():
    llm = init_llm()
    structured_llm_grader = llm.with_structured_output(GradeAnswer)
    return get_answer_grade_prompt() | structured_llm_grader