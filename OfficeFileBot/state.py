import json
import os

# 初始狀態變數
context = "以下是與AI助理的對話。 助理樂於助人、有創意、聰明且非常友善。 " \
          " \n\n人類：你好，你是誰？ 今天我能為您提供什麼幫助？ "

conversation = {
    "Conversation": [
        "你是誰?", "我是辦公室檔案小助手, 可以回答檔案內容"
    ]
}

current_user_message = ""
past_conversations = []
selected_conv = None
selected_row = [1]
content = ""
chunk_size = 500
chunk_overlap = 0
chain = None
skiprows = None
separators = "'\\n\\n', '\\n', ' ', ''"
is_csv = False

def load_config(config_file="config.json"):
    if os.path.exists(config_file):
        with open(config_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

config = load_config()