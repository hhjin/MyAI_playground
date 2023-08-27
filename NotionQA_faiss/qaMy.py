"""Ask a question to the notion database."""
import faiss
from langchain.llms import AzureOpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
import pickle


# Load the LangChain.

# 在Cursor 中运行时， file_path 相对路径是 从当前workspace根目录(py)
pklName ="indexs_backup/faiss_storeStartup.pkl"
indexFile="indexs_backup/notionStartup.index"

print ("load index file : "+indexFile)
index = faiss.read_index(indexFile)

print ("load pkl store file : "+pklName)
with open(pklName, "rb") as f:
    store = pickle.load(f)

store.index = index

llmAzure = AzureOpenAI(
    deployment_name="gpt35turbo",
    #model_name="gpt-3.5-turbo",
    temperature= 0.26
)

llmOpenAI = ChatOpenAI(
    
    model_name="gpt-3.5-turbo-0613",
    temperature= 0.26,
)

# ##Choose a LLM : 
llm=AzureOpenAI


# print(llm ("What is your name?"))
for i in range(500):
    question = input("\n Input your mesage for CHAT-GPT :")
    chain = RetrievalQAWithSourcesChain.from_chain_type( llmAzure , retriever=store.as_retriever())
    result = chain({"question": question})
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

#def printDocs (  List[Tuple[Document, float]] docs) :
    

