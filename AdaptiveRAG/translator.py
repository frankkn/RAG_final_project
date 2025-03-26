from langchain_openai import AzureChatOpenAI
from config import get_model_config
from langchain_core.prompts import ChatPromptTemplate
import language_tool_python
from langchain_core.tools import Tool

grammar_tool = language_tool_python.LanguageTool('en-US')
grammar_tool.disable_spellchecking()

TECH_TERMS = {
    "Enter", "ESC", "F1", "F2", "F3", "F4", "Arrows", "+/-", "<K>", "<M>", "\n"
}

def check_grammar(text):
    """
    檢查文本的文法（不檢查拼字），返回是否有錯誤以及修正後的文本。
    返回值：(是否有錯誤, 原始文本, 修正後文本)
    """
    matches = grammar_tool.check(text)
    if not matches:  # 沒有錯誤
        return False, text, text
    
    # 過濾掉技術術語相關的錯誤
    filtered_matches = [m for m in matches if m.matchedText not in TECH_TERMS]
    if not filtered_matches:  # 過濾後無錯誤
        return False, text, text
    
    # 若有非技術術語的錯誤，進行修正
    corrected_text = grammar_tool.correct(text)
    return True, text, corrected_text

def init_translator(model_version="gpt-4o"):
    config = get_model_config(model_version)
    return AzureChatOpenAI(
        model=config['model_name'],
        deployment_name=config['deployment_name'],
        openai_api_key=config['api_key'],
        openai_api_version=config['api_version'],
        azure_endpoint=config['api_base'],
        temperature=0.0
    )

def translate_text(text, target_language):
    """
    將給定的英文文本翻譯成目標語言。
    """
    llm = init_translator()
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
        你是一位專業的技術翻譯專家，專注於 BIOS 設定的技術術語翻譯。
        請將以下英文文本翻譯成指定的目標語言，確保翻譯準確且符合技術術語的慣用表達。
        目標語言：{target_language}
        """),
        ("human", "請翻譯以下文本：\n\n{text}")
    ])
    response = llm.invoke(prompt.format(target_language=target_language, text=text))
    return response.content.strip()

def translate_missing_fields(data_list, languages=None):
    """
    為 data_list 中的每個 ASUS Token 補全缺失的語言翻譯。
    邏輯：
    1. 只處理 Lost_String 分頁。
    2. 檢查 en-US 是否為空，若為空則跳過該行。
    3. 若 en-US 不為空，則為所有缺失的語言欄位進行翻譯（若該語言已有值則跳過）。
    """
    if languages is None:
        languages = ["zh-cht", "zh-chs", "uk-UA", "es-ES", "ru-RU", "pt-PT", "ko-KR", "ja-JP", "de-DE", "fr-FR"]
    
    updated_data_list = []
    for data in data_list:
        # 1. 只處理 Lost_String 分頁
        if data["metadata"]["sheet"] != "Lost_String":
            updated_data_list.append(data)
            continue
        
        updated_data = data.copy()
        
        # 2. 檢查 en-US 是否為空
        en_us_text = updated_data.get("en-US", "")
        if not en_us_text or en_us_text == "nan":  # 檢查是否為空字符串或 "nan"
            print(f"Warning: Row {data['metadata']['row']} in {data['metadata']['sheet']} has no en-US translation, skipping.")
            updated_data_list.append(updated_data)
            continue
        
        # 3. 獲取 row 號碼（從 metadata 中提取）
        row_number = data["metadata"]["row"]
        has_errors, original_text, corrected_text = check_grammar(en_us_text)
        if has_errors:
            print(f"Revise Row {row_number} en-US from '{original_text}' to '{corrected_text}'")
            en_us_text = corrected_text
        
        # 4. 為所有缺失的語言欄位進行翻譯（若已有值則跳過）
        for lang in languages:
            # 檢查該語言欄位是否為空（包括空字符串或 "nan"）
            if lang not in updated_data or not updated_data[lang] or updated_data[lang] == "nan":
                print(f"Translating Row {row_number} from en-US to {lang}...")
                try:
                    translation = translate_text(en_us_text, lang)
                    updated_data[lang] = translation
                except Exception as e:
                    print(f"Error translating Row {row_number} to {lang}: {str(e)}")
                    updated_data[lang] = ""
            else:
                print(f"Skipping {lang} for Row {row_number} as it already has a translation.")
        
        updated_data_list.append(updated_data)
    
    return updated_data_list