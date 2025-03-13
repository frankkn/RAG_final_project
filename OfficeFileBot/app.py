from taipy.gui import Gui
from loaders import FileLoader
from rag import RAG, csv_file
from ui import on_init, request, update_context, send_message, style_conv, on_exception, reset_chat, tree_adapter, select_conv, page
from state import *

if __name__ == "__main__":
    gui = Gui(page=page)
    gui.on_init = on_init
    gui.run(margin="0em", port=8080, host='0.0.0.0', title="辦公室檔案問答機器人")