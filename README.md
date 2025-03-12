# RAG_FINAL_PROJECT #

這個專案是建立一個基於 Taipy GUI 的 辦公室檔案問答機器人，使用者可以上傳各類型的檔案（如 PDF、Word、Excel、CSV 等），並讓 AI 根據檔案內容提供問答服務。此系統的核心功能包括檔案解析、段落分割、向量化處理及基於 RAG（Retriever-Augmented Generation）技術進行對話生成。使用者可以通過界面與 AI 進行互動，並能夠查看過去的對話紀錄。

## 主要功能

1. 檔案上傳：使用者可以上傳多種檔案格式（.pdf, .docx, .pptx, .csv, .xlsx），系統會根據檔案格式進行解析和處理。

2. 檔案解析：系統支持處理多種文件格式，將文件內容提取並進行段落分割和向量化處理，以便進行問答。

3. 段落分割與向量化處理：上傳的檔案會根據使用者設置的分割字串和重複字元數進行段落分割，並生成向量表示，方便後續檢索和回答。

4. CSV 檔案處理：如果上傳的是 CSV 檔案，使用者可以設置忽略的行數，系統會自動跳過指定的行數並加載檔案。

5. 過去對話記錄：系統會記錄所有對話並支持選擇查看過去的對話記錄，方便使用者回顧。

6. 即時問答：使用者可以在訊息框輸入問題，AI 會根據當前檔案內容及對話上下文提供回答。

## 使用技術

* Taipy GUI：用於構建前端界面，實現與使用者的交互。
* RAG 技術：用於檔案內容的檢索與生成，幫助AI根據檔案提供智能回應。
* Python 函式庫：包括處理各種檔案的函式庫，如 pdf_load 解析 PDF 檔案，office_file 解析 Office 格式檔案，pandas_agent 處理 CSV 檔案等。

## 安裝與運行

```
pip install -r requirements.txt
```

```
cd OfficeFileBot
python app.py
```

If it occurs error like below:

```
ImportError: failed to find libmagic.  Check your installation
```

Uninstall python-magic

```
pip uninstall python-magic
```

## 預覽界面

* 使用者可以通過直覺式界面上傳檔案，設置分割字串和重複字元數，並即時發送訊息進行詢問。
* 可查看過去的對話並選擇進行查看或繼續。

## 注意事項

* 當上傳檔案後，請確保選擇正確的檔案格式，並根據檔案的內容設置適當的分割參數。

## 未來計劃

* 支持多檔案同時上傳與分析。
* 支持更多檔案格式。