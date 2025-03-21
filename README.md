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

`AdaptiveRAG` 是一個基於 RAG（Retrieval-Augmented Generation）技術的技術文件問答系統，專為處理 ASUS 的 2025_ML_UNI 多語言字串翻譯表文件設計。它針對 BIOS 設定的技術術語、產品代碼（如 Gaming、Commercial）和多語言支援提供專業問答，並具備自動翻譯功能。系統結合向量資料庫檢索、網路搜尋和直接回答功能，通過動態路由和品質驗證機制確保回應的準確性和實用性。

### 主要功能

1. **BIOS 字串問答**：針對 2025_ML_UNI 文件，提供與 BIOS 字串翻譯、產品代碼和更新規則相關的專業解答。
2. **動態問題路由**：根據問題類型，自動選擇以下路徑：
   - **向量資料庫檢索**：用於 BIOS 字串翻譯、產品代碼管理和多語言支援相關問題。
   - **網路搜尋**：用於非 BIOS 相關的通用問題。
   - **直接回答**：用於簡單或無需檢索的問題。
3. **文件評分與驗證**：
   - 檢索到的文件經過相關性評分，確保與問題匹配。
   - 生成的答案通過幻覺檢測（hallucination check）和有用性評估，確保基於文件且有效。
4. **持久化資料庫**：向量資料庫儲存於 `./chroma_db`，支持高效查詢且無需每次重建。
5. **自動翻譯功能**：針對 `Lost_String` 分頁，根據英文（en-US）內容補全缺失的語言翻譯，並將結果儲存至新檔案。
6. **支援範例問題**：
   - 「如何將某個 ASUS Token 翻譯成日文？」
   - 「Gaming 產品代碼有哪些多語言支援？」
   - 「什麼是 AMI Token 的作用？」
   - 非 BIOS 問題（如「今天的日期是什麼？」）會轉向網路搜尋或直接回答。

### 使用技術

* **LangChain & LangGraph**：構建 RAG 工作流程，實現檢索、生成、路由和品質驗證邏輯。
* **Chroma 向量資料庫**：儲存和檢索文件嵌入（embeddings），支持高效相似性搜索。
* **Azure OpenAI**：
  - 嵌入生成：使用 `text-embedding-ada-002` 模型。
  - 語言模型：使用 `gpt-4o` 進行問答和翻譯。
* **Tavily Search**：提供網路搜尋功能，補充本地文件無法回答的問題。
* **Python 函式庫**：
  - `pandas`：處理 Excel 文件（如 2025_ML_UNI）。
  - `os` 和 `dotenv`：管理環境變數和檔案路徑。

### 安裝與運行

```
cd AdaptiveRAG
python main.py
```

**第一次運行**：
- 程式會載入 `./example/2025_ML_UNI_20250311.xlsx`，並建立持久化向量資料庫（儲存於 `./chroma_db`）。
- 後續運行將直接使用現有資料庫，提升啟動速度。

**運行選項**：
- 輸入 `1`：提問問題並獲得回應。
- 輸入 `2`：自動翻譯 2025_ML_UNI 文件中 `Lost_String` 分頁的缺失欄位，並儲存至新檔案（如 `2025_ML_UNI_20250311_translated.xlsx`）。
- 輸入 `3` 或 `exit`：結束程式。

### 注意事項

* 確保 `./example/2025_ML_UNI_20250311.xlsx` 檔案存在，否則程式會報錯並拋出 `FileNotFoundError`。
* 系統針對 BIOS 字串翻譯和產品代碼管理問題最佳化，非相關問題可能依賴網路搜尋，回答品質受網路資料影響。
* 自動翻譯功能僅處理 `Lost_String` 分頁，且需 en-US 欄位有值，否則該行會被跳過並顯示警告。

### 未來計劃

* 支持多檔案同時上傳與分析，提升處理能力。
* 優化網路搜尋結果的篩選邏輯，提高非文件問題的回答品質。
* 開發圖形使用者界面（GUI）或 Web 界面，改善使用者體驗。
* 增加對更多語言和文件格式的支持，例如 PDF 或 Word 文件。