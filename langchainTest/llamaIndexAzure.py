# 设置环境变量
import os
os.environ["OPENAI_API_KEY"] = "bed4b05094014794adb4e948ed07d6a0"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://bttaidoc.openai.azure.com/"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"

# 使用Azure OpenAI Service
from llama_index.llms import AzureOpenAI
llm = AzureOpenAI(engine="gpt35turbo-16k", model="gpt-3.5-turbo-16k", temperature=0.0, azure_endpoint="https://bttaidoc.openai.azure.com/", api_key="bed4b05094014794adb4e948ed07d6a0", api_version="2023-07-01-preview")

# 使用完整的端点进行文本完成
#response = llm.complete("The sky is a beautiful blue and")
#print(response)

# 使用聊天端点进行对话
from llama_index.llms import ChatMessage
messages = [
    ChatMessage(role="system", content="You are a pirate with colorful personality."),
    ChatMessage(role="user", content="Hello"),
]
response = llm.chat(messages)
print(response)