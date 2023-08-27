from langchain.llms import OpenAI
from langchain.docstore.document import Document
import requests

import pathlib
import subprocess
import tempfile
from langchain import PromptTemplate ,LLMChain


import sys
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 

##########################################  

myQAKit=QA_Toolkit("./Chroma_DB_300.2300")

llmAzure=myQAKit.get_llm_azure(streaming=True,temperature=0.6, max_tokens=1300, model_name="gpt_3.eeww5_turbo")
storeOpenAI=myQAKit.get_dbstore_openai()

llmchat=myQAKit.get_chat_azure(streaming=True,deployment_name="gpt35turbo-16k", max_tokens=1300, temperature=0.23,)

prompt_template = """Use the context below to write a 333 word blog post about the topic below:
    Context: {context}
    Topic: {topic}
    Blog post:"""

PROMPT = PromptTemplate(   template=prompt_template, input_variables=["context", "topic"]  )

chain = LLMChain(llm=llmchat, prompt=PROMPT)

def generate_blog_post(topic):
    docs = storeOpenAI.similarity_search(topic, k=6)

    myQAKit.printMatchedDocs(docs)
    '''  
    # prompt input approch 1
    inputs = [{"context": doc.page_content, "topic": topic} for doc in docs]
    print("\n\n\n #################### chain.apply 【list】批量生成  AIGC 内容 ：\n\n" ,
     chain.apply(inputs))
    '''   

    # prompt input approch 2
    contexs=[{f"context {i} :": doc.page_content} for i, doc in enumerate(docs)]
    print("\n\n######## 上下文  \n：", contexs)
  
    inputs= {"context": contexs, "topic":topic}
    print("\n\n\n\n #################### chain.run(queyinput ) 生成  AIGC 内容 ：\n\n")
    print(chain.run(inputs))

    
#generate_blog_post("Use UDTT for Multiple Channels Integration of Bank applications")
generate_blog_post("怎么使用BTT加速开发？ Answer in Chinese.")