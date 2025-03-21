from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import UnstructuredFileLoader
import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# def load_and_split_documents(file_path, chunk_size=1024, chunk_overlap=256):
#     if file_path.lower().endswith('.pdf'):
#         loader = PyPDFLoader(file_path=file_path, extract_images=True)
#     elif file_path.lower().endswith(('.docx', '.pptx', '.xlsx')):
#         loader = UnstructuredFileLoader(file_path)

#     splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     return loader.load_and_split(splitter)

# def load_and_split_documents(file_path, chunk_size=5000, chunk_overlap=0):
#     print(f"Loading Excel file: {file_path}")
#     try:
#         xls = pd.ExcelFile(file_path)
#         # print(f"Available sheets: {xls.sheet_names}")
        
#         documents = []
#         # 跳過 'rpl_uni (舊)' 和 '使用說明'
#         for sheet_name in xls.sheet_names:
#             if sheet_name in ['rpl_uni (舊)', '使用說明']:
#                 # print(f"Skipping sheet: {sheet_name} as per request")
#                 continue
            
#             # print(f"Processing sheet: {sheet_name}")
#             try:
#                 df = pd.read_excel(file_path, sheet_name=sheet_name)
#                 # print(f"Loaded {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
                
#                 for index, row in df.iterrows():
#                     num_columns = len(row)
                    
#                     product = str(row.iloc[0]) if num_columns > 0 and pd.notna(row.iloc[0]) else ""
#                     asus_token = str(row.iloc[1]) if num_columns > 1 and pd.notna(row.iloc[1]) else ""
#                     ami_token = str(row.iloc[2]) if num_columns > 2 and pd.notna(row.iloc[2]) else ""
#                     remark = str(row.iloc[14]) if num_columns > 14 and pd.notna(row.iloc[14]) else ""
                    
#                     translations = {}
#                     language_columns = {
#                         3: "en-US", 4: "zh-cht", 5: "zh-chs", 6: "uk-UA", 7: "es-ES",
#                         8: "ru-RU", 9: "pt-PT", 10: "ko-KR", 11: "ja-JP", 12: "de-DE",
#                         13: "fr-FR"
#                     }
#                     for col_idx, lang_code in language_columns.items():
#                         if num_columns > col_idx:
#                             translation = str(row.iloc[col_idx]) if pd.notna(row.iloc[col_idx]) else ""
#                             if translation:
#                                 translations[lang_code] = translation
                    
#                     content = (
#                         f"Sheet: {sheet_name}, Product: {product}, ASUS Token: {asus_token}, "
#                         f"AMI Token: {ami_token}, Remark: {remark}, Translations: {translations}"
#                     )
#                     documents.append(Document(page_content=content, metadata={"source": file_path, "row": index, "sheet": sheet_name}))
#             except Exception as e:
#                 print(f"Error loading sheet {sheet_name}: {str(e)}")
#                 continue
        
#         print(f"Created {len(documents)} documents")
#         # 印出第一筆 Document
#         # if documents:
#         #     print("Sample first document:")
#         #     print(f"Content: {documents[0].page_content}")
#         #     print(f"Metadata: {documents[0].metadata}")
        
#         # 分割文件
#         splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#         return splitter.split_documents(documents)
#     except Exception as e:
#         print(f"General error loading Excel file: {str(e)}")
#         raise

def load_excel_data(file_path):
    """
    載入 Excel 文件，並將數據轉換為字典格式。
    """
    print(f"Loading Excel file: {file_path}")
    try:
        xls = pd.ExcelFile(file_path)
        print(f"Available sheets: {xls.sheet_names}")
        
        data_list = []
        for sheet_name in xls.sheet_names:
            if sheet_name in ['rpl_uni (舊)', '使用說明']:
                print(f"Skipping sheet: {sheet_name} as per request")
                continue
            
            print(f"Processing sheet: {sheet_name}")
            try:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                print(f"Loaded {sheet_name} with {len(df)} rows and {len(df.columns)} columns")
                
                for index, row in df.iterrows():
                    num_columns = len(row)
                    
                    data = {
                        "Product": str(row.iloc[0]) if num_columns > 0 and pd.notna(row.iloc[0]) else "",
                        "ASUS Token": str(row.iloc[1]) if num_columns > 1 and pd.notna(row.iloc[1]) else "",
                        "AMI Token": str(row.iloc[2]) if num_columns > 2 and pd.notna(row.iloc[2]) else "",
                    }
                    
                    language_columns = {
                        3: "en-US", 4: "zh-cht", 5: "zh-chs", 6: "uk-UA", 7: "es-ES",
                        8: "ru-RU", 9: "pt-PT", 10: "ko-KR", 11: "ja-JP", 12: "de-DE",
                        13: "fr-FR"
                    }
                    for col_idx, lang_code in language_columns.items():
                        if num_columns > col_idx:
                            translation = str(row.iloc[col_idx]) if pd.notna(row.iloc[col_idx]) else ""
                            data[lang_code] = translation
                    
                    data["Remark"] = str(row.iloc[14]) if num_columns > 14 and pd.notna(row.iloc[14]) else ""
                    data["metadata"] = {"source": file_path, "row": index, "sheet": sheet_name}
                    
                    data_list.append(data)
            except Exception as e:
                print(f"Error loading sheet {sheet_name}: {str(e)}")
                continue
        
        print(f"Created {len(data_list)} data entries")
        if data_list:
            print("Sample first data entry:")
            print(data_list[0])
        
        return data_list
    except Exception as e:
        print(f"General error loading Excel file: {str(e)}")
        raise

def save_excel_data(file_path, data_list, output_path):
    """
    將更新後的數據保存到新的 Excel 文件中。
    """
    print(f"Saving updated data to: {output_path}")
    try:
        xls = pd.ExcelFile(file_path)
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(file_path, sheet_name=sheet_name)
                if sheet_name == "Lost_String":
                    for data in data_list:
                        if data["metadata"]["sheet"] == "Lost_String":
                            row_idx = data["metadata"]["row"]
                            language_columns = {
                                "en-US": 3, "zh-cht": 4, "zh-chs": 5, "uk-UA": 6, "es-ES": 7,
                                "ru-RU": 8, "pt-PT": 9, "ko-KR": 10, "ja-JP": 11, "de-DE": 12,
                                "fr-FR": 13
                            }
                            for lang, col_idx in language_columns.items():
                                if lang in data and data[lang]:
                                    df.iloc[row_idx, col_idx] = data[lang]
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"Successfully saved updated data to: {output_path}")
    except Exception as e:
        print(f"Error saving Excel file: {str(e)}")
        raise