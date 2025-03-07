# RAG_FINAL_PROJECT #

這個專案是建立一個基於 Taipy GUI 的 辦公室檔案問答機器人，使用者可以上傳各類型的檔案（如 PDF、Word、Excel、CSV 等），並讓 AI 根據檔案內容提供問答服務。此系統的核心功能包括檔案解析、段落分割、向量化處理及基於 RAG（Retriever-Augmented Generation）技術進行對話生成。使用者可以通過界面與 AI 進行互動，並能夠查看過去的對話紀錄。

## 安裝與運行

```
pip install -r requirements.txt
```

```
執行 main.py
```

如果遇到以下error

```
ImportError: failed to find libmagic.  Check your installation
```

請刪除 python-magic

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