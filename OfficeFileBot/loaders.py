# loaders.py
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.document_loaders import PyPDFLoader

class FileLoader:
    """檔案載入器工廠類別，用於根據檔案類型選擇適當的載入方式。"""

    @staticmethod
    def load(file_path):
        """
        根據檔案路徑載入文件。
        Args:
            file_path (str): 檔案的完整路徑
        Returns:
            list: 載入的文件內容（Document 物件列表）
        Raises:
            ValueError: 如果檔案格式不受支援
            Exception: 如果載入過程失敗
        """
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path=file_path, extract_images=True)
                docs = loader.load()
                return docs
            elif file_path.lower().endswith(('.docx', '.pptx', '.xlsx')):
                loader = UnstructuredFileLoader(file_path)
                docs = loader.load()
                return docs
            else:
                raise ValueError(f"不受支援的檔案格式: {file_path}")
        except Exception as e:
            raise Exception(f"檔案載入失敗: {str(e)}")