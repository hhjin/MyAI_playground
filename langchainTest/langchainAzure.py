import os
 
from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage
import langchain
langchain.debug = True

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] =  "2022-12-01"
os.environ["OPENAI_API_BASE"] = "https://bttaidoc.openai.azure.com/"
os.environ["OPENAI_API_KEY"] ='de1c603ee9a84d3aa0c0b82ccbdde577'

##################  Completion API of  Azure OpenAI  


llm = AzureOpenAI(
    deployment_name="gpt35turbo",
    model_name="text-davinci-002",
    temperature= 0,
    max_tokens=26,
)

# Run the LLM

print("\n\n\n#####################  AzureOpenAI   \n")
print(llm)
print()
print(llm("Tell me a joke"))



##################  Chat API of  Azure OpenAI  

chatAzure = AzureChatOpenAI(
    openai_api_base= "https://bttaidoc.openai.azure.com/",
    openai_api_version="2023-03-15-preview",
    deployment_name='gpt35turbo',
    openai_api_key=os.getenv("OPENAI_API_KEY_azure"),
    openai_api_type="azure",
)

print("\n\n\n########################  AzureChatOpenAI  \n")
print(chatAzure)
print()
response= chatAzure(
        [
            HumanMessage(
                content="Tell me a joke"
            )
        ]
)
print(response.content)
