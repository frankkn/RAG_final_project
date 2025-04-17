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
    如果問題要求將某個詞彙翻譯成特定語言：
    - 若文件中包含該詞彙的直接翻譯（例如 'Power Saving Mode'），提取並回答，例如：「根據文件，'Power Saving Mode' 的日文 (ja-JP) 翻譯為『省電力モード』。」
    - 若文件中無完全匹配的詞彙，但包含語義相近的詞彙（例如 'Power Saving' 翻譯為『省電力』，'Mode' 翻譯為『モード』），則根據語義推斷並回答，例如：「根據文件中語義相近的詞彙推斷，'Power Saving Mode' 的日文 (ja-JP) 翻譯為『省電力モード』。」
    - 若文件中無相關詞彙，但 Web Search 結果中提供了相關翻譯，則使用 Web Search 結果並回答，例如：「根據 Web Search 結果，'Power Saving Mode' 的日文 (ja-JP) 翻譯為『省電力モード』。」
    - 若文件和 Web Search 結果均無相關資訊，則回覆：「根據 2025_ML_UNI 文件和 Web Search 結果，我無法回答此問題。」
    特別注意：若進行語義推斷，必須明確註明推斷來源。
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
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "文件: \n\n {document} \n\n 使用者問題: {question}")
    ])

def get_hallucination_grade_prompt():
    instruction = """
    你是一個評分人員，負責確認 LLM 的回應是否虛構。
    以下提供 2025_ML_UNI 文件內容（多語言字串翻譯表，包含 BIOS 設定的技術術語和字串管理規則）與對應的 LLM 回應。請檢查回應是否基於文件內容。
    若回應滿足以下任一條件，則視為基於文件，輸出 'no'：
    - 回應直接引用文件中的翻譯（例如 'Power Saving Mode' 翻譯為『省電力モード』）。
    - 回應基於文件中語義相近的詞彙進行推斷（例如文件中包含 'Power Saving' 翻譯為『省電力』，'Mode' 翻譯為『モード』，則推斷 'Power Saving Mode' 為『省電力モード』）。
    若回應完全未基於文件內容（例如虛構翻譯或無相關詞彙支持），則視為虛構，輸出 'yes'。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "文件: \n\n {documents} \n\n LLM 回應: {generation}")
    ])

def get_answer_grade_prompt():
    instruction = """
    你是一個評分人員，負責評估生成的回答是否對使用者問題有用。
    問題可能涉及技術術語翻譯（例如 'Power Saving Mode' 的日文翻譯）或文件支援的語言列表。
    若回答滿足以下任一條件，則評為有用，輸出 'yes'：
    - 回答提供了問題中詞彙的直接翻譯（例如 'Power Saving Mode' 翻譯為『省電力モード』）。
    - 回答基於文件中語義相近的詞彙進行推斷，並註明推斷來源（例如 '根據文件中語義相近的詞彙推斷，Power Saving Mode 的日文翻譯為『省電力モード』'）。
    - 回答基於 Web Search 結果提供了合理的翻譯，並註明來源。
    - 回答提供了與問題相關的部分資訊（例如列出部分語言列表或與語言支援相關的內容），即使不完整。
    若回答與問題完全無關、翻譯錯誤或未提供任何相關資訊，則評為無用，輸出 'no'。
    """
    return ChatPromptTemplate.from_messages([
        ("system", instruction),
        ("human", "使用者問題: \n\n {question} \n\n 答案: {generation}")
    ])