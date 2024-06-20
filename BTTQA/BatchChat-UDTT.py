from typing import Any, Dict, List, Optional
from langchain import LLMChain
import sys
import time
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit , save_chat_log_file
import langchain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.openai_functions import create_qa_with_structure_chain
from langchain.chat_models import ChatOpenAI ,AzureChatOpenAI
from pydantic import BaseModel, Field
import pydantic


###########################################################################
#  类似于 JS 版本的 BTTBatchChat.ts , 批量 采集 BTTQA chatLogs
#  - RAG context retrieval采用本地 BGE embeeding + Rerank 
#  - openai function call 生成 further questions + source urls 的json response
#
# 在 BTTQA_Train_Datasets_All_clean （17202条记录） 里有 2332 条记录在 Questions_VectorDB_new 里没有
# 在 Questions_VectorDB_new （26122条记录） 里 有 11525 条记录在 datasets 里没有， 两者共有 14870条
#
#   \[[2-9]+\]\n    检查 可用的index数量
############################################################################

class MyCustomResponseSchema(BaseModel):
    """You are a helpful assistant on IT area. When given CONTEXT you answer questions using only that information. You always format your output in markdown. After your answer, you act as an user instead of assistant by asking three specific insightful further questions 
  about the provided CONTEXT to identify the potential questions user want to know. If the CONTEXT includes source URLs you include them under a SOURCES heading before potential questions. The source URLs are indexed with regular express "/-\[\\d+\] : /".The index number is also put in the place of your answer text which is related with the source URL index. 
  Always include all of the relevant source urls from the CONTEXT, but never list a URL more than once (ignore trailing forward slashes when comparing for uniqueness). Never include URLs that are not in the CONTEXT sections. Never make up URLs."""

    Answer :str = Field(..., description="""You answer in markdown format of multi-levels of key points and empty lines between them. You always answer with the content in the CONTEXT provided. You should NOT make up code nonexistent in CONTEXT. You should NOT answer with contents which are not in the CONTEXT provided.""")
    Further_Questions: List[str] = Field(..., description="""Further questions about the provided CONTEXT to identify the potential questions user want to know""")
    Source_URLs: List[str] = Field(..., description="""The source URLs of the contex for the answer content""")
   
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

system_prompt_template = """You are a helpful assistant on IT area. When given CONTEXT you answer questions using only that information. You always format your output in markdown. After your answer, you act as an user instead of assistant by asking three specific insightful further questions 
  about the provided CONTEXT to identify the potential questions user want to know. You include code snippets if relevant. If you are unsure and the answer is not explicitly written in the CONTEXT provided, you say
  \"Sorry, I don't know how to help with that.\". Use the text provided to form your answer, but avoid copying word-for-word from the context text. You try to use your own words when possible."""
system_prompt_template_postfix ="""\n You put the source URL index at the place of the answer text which is related with the source URL index. Never include URLs that are not in the CONTEXT sections. Never make up URLs."""

example_human_template="""
 CONTEXT:
  Next.js is a React framework for creating production-ready web applications. It provides a variety of methods for fetching data, a built-in router, and a Next.js Compiler for transforming and minifying JavaScript code. It also includes a built-in Image Component and Automatic Image Optimization for resizing, optimizing, and serving images in modern formats.
 SOURCE: nextjs.org/docs/faq
  
 USER QUESTION: 
  what is nextjs?"""
example_human = HumanMessagePromptTemplate.from_template(example_human_template)
example_ai_template="""Next.js is a framework for building production-ready web applications using React. It offers various data fetching options, comes equipped with an integrated router, and features a Next.js compiler for transforming and minifying JavaScript. -[1] 
 Additionally, it has an inbuilt Image Component and Automatic Image Optimization that helps resize, optimize, and deliver images in modern formats.-[2]


 Sources:
  -[1]: https://nextjs.org/docs/faq
  -[2]: https://nextjs.org/docs/Image-Component

 Further Questions:
  - Q1: How does next.js built-in Image Component work ?
  - Q2: What is architecture of next.js ?
  - Q3: How to develop page flow with next.js?"""
example_ai = AIMessagePromptTemplate.from_template(example_ai_template)

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template + system_prompt_template_postfix)

human_template = """
CONTEXT:
{context}

-----
USER QUESTION: {query} 
You answer in markdown format of multi-levels of key points and empty lines between them. You always answer with the content in the CONTEXT provided. You should NOT make up code nonexistent in CONTEXT. You should NOT answer with contents which are not in the CONTEXT provided.
You put further questions about the provided CONTEXT to identify the potential questions user want to know at answer end.
At answer end, you list full URLs of Sources which are referenced by answer. You remember to put index number of URLs list at the right place of the answer's main body which are referenced from source URL. Never include URLs that are not in the CONTEXT sections. Never make up URLs.
"""
 
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template ,input_variables=["context", "query"] )
chat_prompt = ChatPromptTemplate.from_messages(
    ### with one shot in context learnning
    [system_message_prompt, example_human, example_ai, human_message_prompt]

     ### No in context learnning
    #[system_message_prompt,  human_message_prompt]
)

def bttqa_chat_chain(llm, store, query , chat_prompt, topk=6 , rerank=False, funcall=False ,rootPath=""):
    print(f"\n\n###################   BTT qa_with_structure_chain,  store : {store} \n\n####### LLM : {llm} \n  ####### topk={topk}  rerank={rerank}   funcall :{funcall}")
    docs = myQAKit.similarity_search(store, query,printDocs=False, k=topk) 
    if rerank:
        contexs = myQAKit.reRanker_BGE_large(query, docs ,topk=7)
        contexs_text=""
        for doctxt in contexs:
            contexs_text+=doctxt
    else:
        #contexs=[{f"\n### context {i} :": doc.page_content.strip(), "\n### Source: " : doc.metadata["source"] } for i, doc in enumerate(docs)] 
        contexs_text=""
        for i, doc in enumerate(docs) :
          contexs_text=contexs_text+  f"\n{doc.page_content.strip()}\n## Source: {doc.metadata['source']}\n---------\n"
       
    print(f"\n\n\n ############  contexs_text : \n{contexs_text} ")
    inputs= {"context": contexs_text, "query":query.strip()}
    print("\n\n#### AI response:  ---->  ¡™£¢∞§¶•ªº–≠œ∑´®†¥˙©ƒ∆˚¬…æµ˜∫ç∫√ç≈≈≥÷≤µ˜∫√√˙≈∞§∞§§•¡™£¢∞§¶•ππ©˙∆˚¬…ææ…≈ç√∫≤≥÷   ---->\n")
  
    if funcall :   # function call 
        response = create_qa_with_structure_chain(llm, MyCustomResponseSchema, verbose = True, output_parser="pydantic", prompt=chat_prompt).run( inputs )
        print(f"{response.Answer} \n### Further questions: {response.Further_Questions}  \n###Sources: {response.Source_URLs} ")
        answer=response.Answer
    else :
        qaChatchain = LLMChain(llm=llm, prompt=chat_prompt )
        response=qaChatchain.run(inputs)
        answer=response 
    save_chat_log_file(query, contexs_text, answer , "ic490-bgeEmb-bgeRank", "azure_chat"  , rootPath=rootPath)

    return response


#############     main start   #############################  
 
langchain.verbose = True
 ## default conda env 3.10.9
## 0.4.0 chormadb 版本之前的旧 格式数据库：
myQAKit=QA_Toolkit("LocalData/old_chroma/Chroma_DB_UDTT_IC490")
storeOpenAI_Chroma=myQAKit.get_dbstore_openai( )
storeBGE_Chroma=myQAKit.get_dbstore_BGE( )

# conda env:" llama-index or base on M1 ;   base on Windows 
## new chormadb version DB file migrated from old version created by old langchain, can be used by new langchain
#myQAKit=QA_Toolkit("LocalData/chroma/Chroma_DB_UDTT_IC490_migrated")
#storeOpenAI_Chroma=myQAKit.get_dbstore_openai( )
#storeBGE_Chroma=myQAKit.get_dbstore_BGE( )
#storeBce_Chroma=myQAKit.get_dbstore_bce()

# conda env:" llama-index or base on M1 ;   base on Windows  
## new chormadb version DB file created by llama-index, 注意collection_name 要匹配
#myQAKit=QA_Toolkit("LocalData/chroma/UDTTIC490_llamaindex-AllChunks/openai-ada2-allrandomID")
#storeOpenAI_Chroma=myQAKit.get_dbstore_openai(chromaPath_suffix=False , collection_name="quickstart")

 
# --function call 无法用 stream response   
# --如果设置max_tokens，会检查 request_tokens + max_tokens 是否大于模型最大Context，否则只检查request_tokens
llmAzureChat=AzureChatOpenAI(streaming=True, deployment_name="gpt35turbo-16k", max_tokens=1000, temperature=0 ,callbacks=[StreamingStdOutCallbackHandler()])

'''
#store_Cohere=myQAKit.get_dbstore_cohere()
#store_SETF=myQAKit.get_dbstore_sentence()
#cohere 不支持Chatchain
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llamaCpp = LlamaCpp(
    model_path="/Users/henryking/llama.cpp/models/llama-2-13b-chat.ggmlv3.q5_0.bin",
    #model_path="/Users/henryking/llama.cpp/models/ggml-vic7b-q5_1.bin",
    input={"temperature": 0.5, "max_length": 3000, "top_p": 1},
    n_ctx=2900,
    callback_manager=callback_manager,
    verbose=True,
)
'''

'''
while True:
        query =input("\n\n####### Question for BTT: ")
        if query==" ":    # default query for space input        
          query="What is new in each version of UDTT?"
        else :
          if query=="":
                break
        
        result=bttqa_chat_chain(llmAzureChat, storeBGE_Chroma, query, chat_prompt ,13,rerank=True )               
'''

import json
breakedIndex=1800  # count from 1
with open('noQA_questions.json', 'r', encoding="utf-8") as file:
    noQA_questions = json.load(file)
    
    for i, item in enumerate(noQA_questions) :
        if i < breakedIndex:
           continue
        query=item["question"]
        print(f"##### noQA_questions  index={i+1}    query: {query}")
        if i % 2 == 0  and i % 10 !=0 and i % 14 !=0 :
            try :
                result = bttqa_chat_chain(llmAzureChat, storeBGE_Chroma, query, chat_prompt, 13, rerank=True ,rootPath="LocalData/chatLogs/ic490-bgeEmb-bgeRank_i2-16k")
            except BaseException as e:
                print(f"Error occurred during generation: {e}")



