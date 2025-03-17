from langchain_core.prompts import ChatPromptTemplate

def get_route_prompt():
    instruction = """
    你是將使用者問題導向向量資料庫或網路搜尋的專家。
    向量資料庫包含 ASUS Y2024H2 Intel Platform Commercial BIOS Setup Menu Specification 文件，涵蓋 Intel 平台（AlderLake, RaptorLake, LunarLake, ArrowLake, TwinLake）的 BIOS 設定菜單行為與規格。
    僅對於與 BIOS 設定、硬體配置或相關功能的問題使用向量資料庫工具。所有其他問題（包括日常知識等）一律使用網路搜尋工具。
    """
    return ChatPromptTemplate.from_messages([("system", instruction), ("human", "{question}")])

def get_rag_prompt():
    instruction = """
    你是一位負責處理使用者問題的技術專家，請利用提取出的 ASUS Y2024H2 Intel Platform Commercial BIOS Setup Menu Specification 文件內容來回應問題。
    回答時請使用專業技術術語，並確保內容準確且符合文件規格。若問題的答案無法從文件中取得，請直接回覆「根據文件中提供的資訊，我無法回答此問題」，禁止虛構答案。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("system", "文件內容: \n\n {documents}"),
        ("human", "問題: {question}")
    ])

def get_plain_prompt():
    instruction = """
    你是一位負責處理使用者問題的技術助手，請利用你的知識來回應問題。
    回應時請確保答案的技術準確性，並優先參考 ASUS BIOS 相關的通用知識。若無法確定答案，請說明「我無法提供確切答案」，勿虛構內容。
    """
    return ChatPromptTemplate.from_messages([("system", instruction), ("human", "問題: {question}")])

def get_retrieval_grade_prompt():
    instruction = """
    你是一個評分人員，負責評估文件與使用者問題的關聯性。
    如果文件包含與使用者問題相關的關鍵資訊或語意，則評為相關，輸出 'yes'；否則輸出 'no'。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "文件: \n\n {document} \n\n 使用者問題: {question}")
    ])

def get_hallucination_grade_prompt():
    instruction = """
    你是一個評分人員，負責確認 LLM 的回應是否虛構。
    以下提供 ASUS Y2024H2 Intel Platform Commercial BIOS Setup Menu Specification 文件內容與對應的 LLM 回應。請檢查回應是否基於文件內容。
    輸出 'yes' 表示回應是虛構的，未基於文件內容；'no' 表示回應未虛構，基於文件內容得出。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "文件: \n\n {documents} \n\n LLM 回應: {generation}")
    ])

def get_answer_grade_prompt():
    instruction = """
    你是一個評分人員，負責確認答案是否回應了問題。
    輸出 'yes' 或 'no'。'Yes' 表示答案確實回應了問題，'No' 表示答案未回應問題。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}")
    ])