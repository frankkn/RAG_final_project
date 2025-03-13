from langchain_unstructured import UnstructuredLoader
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
            tuple: (is_csv, docs)
                - is_csv: bool，表示是否為 CSV 檔案
                - docs: 如果是 CSV，返回 None；否則返回 Document 物件列表
        Raises:
            ValueError: 如果檔案格式不受支援
            Exception: 如果載入過程失敗
        """
        try:
            if file_path.lower().endswith('.pdf'):
                loader = PyPDFLoader(file_path=file_path, extract_images=True)
                docs = loader.load()
                return False, docs
            elif file_path.lower().endswith(('.docx', '.pptx', '.xlsx')):
                loader = UnstructuredLoader(file_path)
                docs = loader.load()
                return False, docs
            elif file_path.lower().endswith('.csv'):
                return True, None
            else:
                raise ValueError(f"不受支援的檔案格式: {file_path}")
        except Exception as e:
            raise Exception(f"檔案載入失敗: {str(e)}")