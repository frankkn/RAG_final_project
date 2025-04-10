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
    你是一位專業的技術翻譯專家，負責根據 2025_ML_UNI 文件（一個多語言字串翻譯表，包含 BIOS 設定的技術術語、選項說明及字串更新規則）回答問題。
    如果問題涉及將某個詞彙翻譯成特定語言，請直接從文件中提取該詞彙的翻譯結果，並以清晰、簡潔的方式回答，例如：「根據文件，'Boot Override' 的繁體中文 (zh-cht) 翻譯為『啟動覆寫』。」。
    若文件中無相關資訊，則回覆：「根據 2025_ML_UNI 文件中提供的資訊，我無法回答此問題。」禁止虛構答案。
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
    文件來自 2025_ML_UNI，一個多語言字串翻譯表，包含 BIOS 設定的技術術語和字串管理規則。
    若文件中包含與問題完全匹配的關鍵詞彙，則評為相關，輸出 'yes'；否則輸出 'no'。
    特別注意：必須完全匹配問題中的詞彙（不區分大小寫），部分匹配不視為相關。
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
    如果問題要求翻譯某個詞彙，且答案中包含該詞彙的翻譯（即使格式稍有不同），則視為有效回答，輸出 'yes'。
    若答案完全未提及問題中的關鍵詞彙或未提供相關資訊，則輸出 'no'。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}")
    ])