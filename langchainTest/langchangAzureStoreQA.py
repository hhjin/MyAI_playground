import os
import openai

from langchain.llms import AzureOpenAI

 
openai.api_type = "azure"
openai.api_base = "https://bttaidoc.openai.azure.com/"
openai.api_version = "2022-12-01"
openai.api_key = 'de1c603ee9a84d3aa0c0b82ccbdde577'

import warnings
warnings.filterwarnings('ignore')

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch

file = 'Tweets.csv'
loader = CSVLoader(file_path=file)
data = loader.load()


# VectorstoreIndexCreator call openAI embedding which get key ONLY from OS variable of .profile
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])



##################  Completion API of  Azure OpenAI  

llm = AzureOpenAI(
    deployment_name="gpt35turbo",
    model_name="text-davinci-002",
    temperature= 0,

)

qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)

examples = [
    {
        "query": "Does Apple published new \
            AR VR devices?",
        "answer": "Yes"
    },
    {
        "query": "How to make use of  \
        AI for personal productivity?",
        "answer": "Make use of OpenAI for work flow and coding"
    }
]

