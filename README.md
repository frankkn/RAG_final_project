<div align="right">

**English** | [繁體中文](README.zh-TW.md)

</div>

# RAG_FINAL_PROJECT

This repository contains two independent sub-projects: `AdaptiveRAG` and `OfficeFileBot`.
- `AdaptiveRAG` is a RAG-based question-answering system for technical documents. It focuses on the ASUS Y2025_ML_UNI multilingual string translation spreadsheet, providing expert Q&A on BIOS settings along with automatic translation.
- `OfficeFileBot` is an office-file Q&A chatbot built on Taipy GUI, supporting question answering over a wide range of file formats.

## 1. AdaptiveRAG

`AdaptiveRAG` is a technical-document Q&A system built on RAG (Retrieval-Augmented Generation), designed specifically for ASUS's 2025_ML_UNI multilingual string translation spreadsheet. It delivers expert answers about BIOS technical terminology, product codes (e.g., Gaming, Commercial), and multilingual support, and it can translate missing strings automatically. The system combines vector-database retrieval, web search, and direct answering, using dynamic routing and quality-verification mechanisms to keep responses accurate and useful.

### Key Features

1. **BIOS string Q&A**: Answers questions about BIOS string translations, product codes, and update rules based on the 2025_ML_UNI document.
2. **Dynamic question routing**: Automatically picks the best path for each question:
   - **Vector-database retrieval**: For questions about BIOS string translation, product code management, and multilingual support.
   - **Web search**: For general questions unrelated to BIOS.
   - **Direct answering**: For simple questions that don't need retrieval.
3. **Document grading and verification**:
   - Retrieved documents are graded for relevance to make sure they match the question.
   - Generated answers go through a hallucination check and a usefulness evaluation, ensuring they are grounded in the documents and actually helpful.
4. **Persistent database**: The vector database is stored in `./chroma_db`, enabling fast queries without rebuilding on every run.
5. **Automatic translation**: For the `Lost_String` sheet, fills in missing language translations based on the English (en-US) content and saves the results to a new file.
6. **Example questions supported**:
   - "How do I translate a given ASUS Token into Japanese?"
   - "Which languages does the Gaming product code support?"
   - "What does an AMI Token do?"
   - Non-BIOS questions (e.g., "What's today's date?") are routed to web search or answered directly.

### Tech Stack

- **LangChain & LangGraph**: Build the RAG workflow, implementing retrieval, generation, routing, and quality-verification logic.
- **Chroma vector database**: Stores and retrieves document embeddings for efficient similarity search.
- **Azure OpenAI**:
  - Embeddings: `text-embedding-ada-002`.
  - Language model: `gpt-4o` for Q&A and translation.
- **Tavily Search**: Provides web search to cover questions the local documents can't answer.
- **Python libraries**:
  - `pandas`: Processes Excel files (e.g., 2025_ML_UNI).
  - `os` and `dotenv`: Manage environment variables and file paths.

### Installation & Usage

```
cd AdaptiveRAG
python main.py
```

**First run**:
- The program loads `./example/2025_ML_UNI_20250311.xlsx` and builds a persistent vector database (stored in `./chroma_db`).
- Subsequent runs reuse the existing database, so startup is much faster.

**Runtime options**:
- Enter `1`: Ask a question and get an answer.
- Enter `2`: Automatically translate the missing fields in the `Lost_String` sheet of the 2025_ML_UNI file and save them to a new file (e.g., `2025_ML_UNI_20250311_translated.xlsx`).
- Enter `3` or `exit`: Quit the program.

### Notes

* Make sure `./example/2025_ML_UNI_20250311.xlsx` exists; otherwise the program raises a `FileNotFoundError`.
* The system is optimized for BIOS string translation and product code management. Unrelated questions may fall back to web search, so answer quality depends on what's available online.
* Automatic translation only processes the `Lost_String` sheet and requires the en-US column to have a value; otherwise the row is skipped with a warning.

### Roadmap

* Support uploading and analyzing multiple files at once.
* Improve the filtering logic for web search results to raise answer quality on non-document questions.
* Build a graphical user interface (GUI) or web interface for a better user experience.
* Add support for more languages and file formats, such as PDF and Word documents.

## 2. OfficeFileBot

`OfficeFileBot` is an office-file Q&A chatbot built on Taipy GUI. Users can upload files of various types (PDF, Word, Excel, CSV, etc.) and have the AI answer questions based on the file contents. Its core capabilities include file parsing, paragraph splitting, vectorization, and conversation generation powered by RAG (Retrieval-Augmented Generation). Users interact with the AI through the interface and can also browse their past conversations.

### Key Features

1. File upload: Users can upload files in multiple formats (.pdf, .docx, .pptx, .csv, .xlsx); the system parses and processes each file according to its format.

2. File parsing: The system handles multiple document formats, extracting content and running paragraph splitting and vectorization so the content is ready for Q&A.

3. Paragraph splitting and vectorization: Uploaded files are split into paragraphs based on the user-configured split string and overlap character count, then converted into vector representations for later retrieval and answering.

4. CSV file handling: For CSV uploads, users can configure how many rows to skip; the system skips the specified rows automatically before loading the file.

5. Conversation history: All conversations are recorded, and users can select and review past conversations at any time.

6. Real-time Q&A: Users type questions in the message box, and the AI answers based on the current file contents and the conversation context.

### Tech Stack

* Taipy GUI: Builds the front-end interface for user interaction.
* RAG: Powers retrieval and generation over file contents, letting the AI respond intelligently based on the files.
* Python libraries: Includes libraries for handling different file types, such as `pdf_load` for parsing PDF files, `office_file` for Office formats, and `pandas_agent` for CSV files.

### Installation & Usage

```
pip install -r requirements.txt
```

```
cd OfficeFileBot
python app.py
```

If you run into the following error:

```
ImportError: failed to find libmagic.  Check your installation
```

uninstall python-magic:

```
pip uninstall python-magic
```

### Interface Preview

* Users can upload files through an intuitive interface, configure the split string and overlap character count, and send questions in real time.
* Past conversations can be browsed and either reviewed or continued.

### Notes

* After uploading a file, make sure you select the correct file format and set split parameters appropriate for its content.

### Roadmap

* Support uploading and analyzing multiple files at once.
* Support more file formats.
