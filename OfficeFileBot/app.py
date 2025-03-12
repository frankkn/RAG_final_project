from taipy.gui import Gui
from ui import page, on_init, send_message, reset_chat, select_conv, style_conv
from rag import RAG, csv_file
from state import *

# 確保所有變數和函數在全局範圍內可見
# Taipy 會自動將這些變數綁定到頁面中使用的名稱
global context, conversation, current_user_message, past_conversations, selected_conv, selected_row
global content, chunk_size, chunk_overlap, chain, separators, skiprows, is_csv

Gui.on_init = on_init

if __name__ == "__main__":
    gui = Gui(page=page)
    gui.run(margin="0em", port=8080, host='0.0.0.0', title="辦公室檔案問答機器人")