from langchain_core.prompts import ChatPromptTemplate

def get_route_prompt():
    instruction = """
    你是將使用者問題導向向量資料庫或網路搜尋的專家。
    向量資料庫包含 2025_ML_UNI 文件，該文件是一個多語言字串翻譯表，涵蓋 BIOS 設定的技術術語、選項說明，以及字串更新和管理規則。
    僅對於與 BIOS 字串翻譯、產品代碼 (如 Gaming、Commercial)、多語言支援或字串管理相關的問題使用向量資料庫工具。所有其他問題（包括非技術性或日常知識問題）一律使用網路搜尋工具。
    """
    return ChatPromptTemplate.from_messages([("system", instruction), ("human", "{question}")])

def get_rag_prompt():
    instruction = """
    你是一位負責處理使用者問題的技術專家，請利用提取出的 2025_ML_UNI 文件內容來回應問題。
    該文件是一個多語言字串翻譯表，包含 BIOS 設定的技術術語、選項說明及字串更新規則。回答時請使用專業技術術語，並確保內容準確且符合文件中的翻譯或規則。若問題的答案無法從文件中取得，請直接回覆「根據 2025_ML_UNI 文件中提供的資訊，我無法回答此問題」，禁止虛構答案。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("system", "文件內容: \n\n {documents}"),
        ("human", "問題: {question}")
    ])

def get_plain_prompt():
    instruction = """
    你是一位負責處理使用者問題的技術助手，請利用你的知識來回應問題。
    回應時請確保答案的技術準確性，並優先參考 2025_ML_UNI 文件中與 BIOS 字串翻譯和產品代碼管理相關的通用知識。若無法確定答案，請說明「我無法提供確切答案」，勿虛構內容。
    """
    return ChatPromptTemplate.from_messages([("system", instruction), ("human", "問題: {question}")])

def get_document_grade_prompt():
    instruction = """
    你是一個評分人員，負責評估文件與使用者問題的關聯性。
    文件來自 2025_ML_UNI，一個多語言字串翻譯表，包含 BIOS 設定的技術術語和字串管理規則。若文件包含與使用者問題相關的關鍵資訊（如字串翻譯、產品代碼或規則），則評為相關，輸出 'yes'；否則輸出 'no'。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "文件: \n\n {document} \n\n 使用者問題: {question}")
    ])

def get_hallucination_grade_prompt():
    instruction = """
    你是一個評分人員，負責確認 LLM 的回應是否虛構。
    以下提供 2025_ML_UNI 文件內容（多語言字串翻譯表，包含 BIOS 設定的技術術語和字串管理規則）與對應的 LLM 回應。請檢查回應是否基於文件內容。
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