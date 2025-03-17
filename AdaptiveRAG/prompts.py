from langchain_core.prompts import ChatPromptTemplate

def get_route_prompt():
    instruction = """
    你是將使用者問題導向向量資料庫或網路搜尋的專家。
    向量資料庫包含有關牙周病治療或照護文件。對於這些主題的問題，請使用向量資料庫工具。其他情況則使用網路搜尋工具。
    """
    return ChatPromptTemplate.from_messages([("system", instruction), ("human", "{question}")])

def get_rag_prompt():
    instruction = """
    你是一位負責處理使用者問題的助手，請利用提取出來的文件內容來回應問題。
    若問題的答案無法從文件內取得，請直接回覆你不知道，禁止虛構答案。
    注意：請確保答案的準確性。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("system", "文件: \n\n {documents}"),
        ("human", "問題: {question}")
    ])

def get_plain_prompt():
    instruction = """
    你是一位負責處理使用者問題的助手，請利用你的知識來回應問題。
    回應問題時請確保答案的準確性，勿虛構答案。
    """
    return ChatPromptTemplate.from_messages([("system", instruction), ("human", "問題: {question}")])

def get_retrieval_grade_prompt():
    instruction = """
    你是一個評分的人員，負責評估文件與使用者問題的關聯性。
    如果文件包含與使用者問題相關的關鍵字或語意，則將其評為相關。
    輸出 'yes' or 'no' 代表文件與問題的相關與否。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "文件: \n\n {document} \n\n 使用者問題: {question}")
    ])

def get_hallucination_grade_prompt():
    instruction = """
    你是一個評分的人員，負責確認LLM的回應是否為虛構的。
    以下會給你一個文件與相對應的LLM回應，請輸出 'yes' or 'no'做為判斷結果。
    'Yes' 代表LLM的回答是虛構的，未基於文件內容 'No' 則代表LLM的回答並未虛構，而是基於文件內容得出。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "文件: \n\n {documents} \n\n LLM 回應: {generation}")
    ])

def get_answer_grade_prompt():
    instruction = """
    你是一個評分的人員，負責確認答案是否回應了問題。
    輸出 'yes' or 'no'。 'Yes' 代表答案確實回應了問題， 'No' 則代表答案並未回應問題。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}")
    ])