from langchain.llms import OpenAI
from langchain.docstore.document import Document
from typing import Any, Dict, List, Optional
from langchain import PromptTemplate ,LLMChain
import sys
import time
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage


#############     main start   #############################  

myQAKit=QA_Toolkit("./Chroma_DB_300.2300")
storeOpenAI_Chroma=myQAKit.get_dbstore_cohere()
llmchat=myQAKit.get_chat_azure(streaming=True,deployment_name="gpt35turbo-16k", max_tokens=111, temperature=0.23,)

prompt_template = """Use the context below to write a 333 word blog post about the topic below:
    Context: {context}
    Topic: {topic}
    Blog post:"""
context="""Next.js is a framework for building production-ready web applications using React. It offers various data fetching options, comes equipped with an integrated router, and features a Next.js compiler for transforming and minifying JavaScript. -[1] 
 Additionally, it has an inbuilt Image Component and Automatic Image Optimization that helps resize, optimize, and deliver images in modern formats.-[2]"""
topic="Next.js"
inputs= {"context": context, "topic":topic}
PROMPT = PromptTemplate(   template=prompt_template, input_variables=["context", "topic"]  )


chain = LLMChain(llm=llmchat, prompt=PROMPT)


chain.run(inputs)


messages = [
    SystemMessage(
        content="You are a helpful assistant that translates Chinese to French."
    ),
    HumanMessage(
        content="Translate this sentence from English to Chinese. I love programming."
    ),
]
llmchat(messages)

