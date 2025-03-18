# RAG_FINAL_PROJECT

這個專案包含兩個獨立的子專案：`OfficeFileBot` 和 `AdaptiveRAG`。  
- `OfficeFileBot` 是一個基於 Taipy GUI 的辦公室檔案問答機器人，支援多種檔案格式的問答服務。  
- `AdaptiveRAG` 是一個基於 RAG 技術的技術文件問答系統，專注於處理 ASUS Y2024H2 Intel Platform Commercial BIOS Setup Menu Specification 文件，提供專業的 BIOS 設定問答。

## 1. OfficeFileBot

`OfficeFileBot` 是一個基於 Taipy GUI 的辦公室檔案問答機器人，使用者可以上傳各類型的檔案（如 PDF、Word、Excel、CSV 等），並讓 AI 根據檔案內容提供問答服務。此系統的核心功能包括檔案解析、段落分割、向量化處理及基於 RAG（Retrieval-Augmented Generation）技術進行對話生成。使用者可以通過界面與 AI 進行互動，並能夠查看過去的對話紀錄。

### 主要功能

1. 檔案上傳：使用者可以上傳多種檔案格式（.pdf, .docx, .pptx, .csv, .xlsx），系統會根據檔案格式進行解析和處理。

2. 檔案解析：系統支持處理多種文件格式，將文件內容提取並進行段落分割和向量化處理，以便進行問答。

3. 段落分割與向量化處理：上傳的檔案會根據使用者設置的分割字串和重複字元數進行段落分割，並生成向量表示，方便後續檢索和回答。

4. CSV 檔案處理：如果上傳的是 CSV 檔案，使用者可以設置忽略的行數，系統會自動跳過指定的行數並加載檔案。

5. 過去對話記錄：系統會記錄所有對話並支持選擇查看過去的對話紀錄，方便使用者回顧。

6. 即時問答：使用者可以在訊息框輸入問題，AI 會根據當前檔案內容及對話上下文提供回答。

### 使用技術

* Taipy GUI：用於構建前端界面，實現與使用者的交互。
* RAG 技術：用於檔案內容的檢索與生成，幫助 AI 根據檔案提供智能回應。
* Python 函式庫：包括處理各種檔案的函式庫，如 `pdf_load` 解析 PDF 檔案，`office_file` 解析 Office 格式檔案，`pandas_agent` 處理 CSV 檔案等。

### 安裝與運行

```
pip install -r requirements.txt
```

```
cd OfficeFileBot
python app.py
```

如果遇到以下錯誤：

```
ImportError: failed to find libmagic.  Check your installation
```

請解除安裝 python-magic：

```
pip uninstall python-magic
```

### 預覽界面

* 使用者可以通過直覺式界面上傳檔案，設置分割字串和重複字元數，並即時發送訊息進行詢問。
* 可查看過去的對話並選擇進行查看或繼續。

### 注意事項

* 當上傳檔案後，請確保選擇正確的檔案格式，並根據檔案的內容設置適當的分割參數。

### 未來計劃

* 支持多檔案同時上傳與分析。
* 支持更多檔案格式。


## 2. AdaptiveRAG 

`AdaptiveRAG` 是一個基於 RAG（Retrieval-Augmented Generation）技術的技術文件問答系統，專為 ASUS Y2024H2 Intel Platform Commercial BIOS Setup Menu Specification 文件設計。它支援 Intel 平台（AlderLake, RaptorLake, LunarLake, ArrowLake, TwinLake）的 BIOS 設定相關問答，並結合向量資料庫檢索和網路搜尋，提供準確且專業的回答。系統會根據問題的性質，選擇使用本地文件檢索、網路搜尋或直接回答，並通過文件評分和答案驗證確保回應品質。

### 主要功能

1. BIOS 文件問答：針對 ASUS Y2024H2 Intel Platform Commercial BIOS Setup Menu Specification 文件，提供專業的 BIOS 設定和硬體配置問答。
2. 動態問題路由：根據問題類型，自動選擇使用向量資料庫檢索（針對 BIOS 相關問題）、網路搜尋（針對通用問題）或直接回答。
3. 文件評分與驗證：
    * 檢索到的文件會經過相關性評分，確保只使用與問題相關的資料。
    * 答案會經過幻覺檢測（hallucination check）和有用性評估，確保回應基於文件且有效。
4. 持久化資料庫：向量資料庫會持久化儲存，避免每次啟動時重建，加快查詢速度。
5. 支援範例問題：
    * 如何設定 Asset Tag？它與 Service Tag 有什麼區別？
    * 哪些 production line 支援 Boot Indicator？
    * 非 BIOS 相關問題（例如「如何治療 PTSD？」）會自動轉向網路搜尋。

### 使用技術

* LangChain & LangGraph：用於構建 RAG 工作流程，實現檢索、生成和路由邏輯。
* Chroma 向量資料庫：儲存和檢索文件嵌入（embeddings），支援高效的相似性搜索。
* Azure OpenAI：提供嵌入生成（text-embedding-ada-002）和語言模型（gpt-4o）功能。
* Tavily Search：用於網路搜尋，補充本地文件無法回答的問題。
* Python 函式庫：包括 PyPDFLoader（PDF 解析）、RecursiveCharacterTextSplitter（文件分割）等。

### 安裝與運行

```
cd AdaptiveRAG
python main.py
```

第一次運行：
程式會載入 ./example/Y2024H2 Intel Platform_Commercial_BIOS_Setup_Menu_Specification_V2.0.7.pdf 並建立持久化資料庫（儲存在 ./chroma_db）。
之後的運行會直接使用已建立的資料庫。

### 注意事項

* 確保 ./example/Y2024H2 Intel Platform_Commercial_BIOS_Setup_Menu_Specification_V2.0.7.pdf 檔案存在，否則程式會報錯。
* 本專案針對 BIOS 相關問題最佳化，其他問題可能會轉向網路搜尋，回答品質可能因網路資料而異。

### 未來計劃

* 支持多檔案同時上傳與分析。
* 優化網路搜尋結果的篩選，進一步提升回答品質。
* 提供更友善的使用者界面（例如 GUI 或 Web 界面）。