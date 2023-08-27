
####################  ###############################################  ############################
#
#    利用openAI 0613 版本 提供的 function 功能， 和 Conversationa/Memory lRetrievalChain  
#    提供结构化的 source urls, further questions, 和conversation history in context
#
#   ### 目前问题：  1. 速度很慢 （function有关） 
#                 2。 没有stream output ，即使设置了，是否设置正确 ,还是function的限制？
#                 3.  condense_question_chain 可以用一个不同的（stream=false)模型instance
#                 4.  condense_question_chain 这种方式过时了，
#                 5.  去掉output_parser="pydantic"好像对结果也没有影响 #只是result输出格式需要自己解析
#                  
#                 6.   可以支持 memory, 但要去掉 pydantic parser, 否则 AIMessage 无法 mapping to memory
#                       但是 memory的 chat_history 会包含 source和 further_question信息
#                 7.  设置 output_parser="pydantic" ，不支持memory 但further_question 和 source是解析好的， 
#                       但需要自己处理 chat_history , 包括 chat_history的格式转换
#                      总之自己需要做一种格式处理， 看需要further_question/source 还是 tokenbuffermemory
#               
#  目前看自己写一个get_chat_history，保留further_question/source ，比较简单，而且能去掉source和 further_questiond对chat history的污染。 
#   缺点就是 tokensize 要自己控制                  
# 
#    
####################  ###############################################  ############################

from typing import Any, Dict, List, Optional
from langchain import PromptTemplate ,LLMChain
import sys
import time
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 
import langchain

from typing import List
from pydantic import BaseModel, Field
from langchain.chains.openai_functions import create_qa_with_structure_chain
from langchain.prompts.chat import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.chains import create_qa_with_sources_chain
from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory ,ConversationTokenBufferMemory

#############     main start   #############################  
langchain.verbose = True
myQAKit=QA_Toolkit("./Chroma_DB_300.2300")
#storeOpenAI_Chroma=myQAKit.get_dbstore_openai()
#storeOpenAI_Supabase=myQAKit.get_dbstore_supabase()
store_Cohere=myQAKit.get_dbstore_cohere()
#store_SETF=myQAKit.get_dbstore_sentence()

#############   选择一个vector store  ###################### 
docsearch = store_Cohere

#llm=myQAKit.get_chat_azure(streaming=True,deployment_name="gpt35turbo-16k", max_tokens=2300, temperature=0.23,)

llm = ChatOpenAI(temperature=0,   model="gpt-3.5-turbo-0613", max_tokens=500, streaming=True ,callbacks=[StreamingStdOutCallbackHandler()])


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.\
Make sure to avoid using any unclear pronouns.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)




class CustomResponseSchema(BaseModel):
    """An answer to the question being asked, with sources."""

    answer: str = Field(..., description="Answer to the question that was asked")
    countries_referenced: List[str] = Field(..., description="All of the countries mentioned in the sources")
    further_questions: List[str] = Field(..., description="Ask three further follow-up questions according to the context sources")
    sources: List[str] = Field(
        ..., description="List of sources used to answer the question"
    )


language="in Chinese"

prompt_messages = [
    SystemMessage(
        content=(
            "You are a world class algorithm to answer "
            "questions in a specific format."
            "After your answer, you act as an user instead of assistant by asking three specific insightful further questions " 
            "about the provided CONTEXT to identify the potential questions user want to know. You include code snippets if relevant. If you are unsure and the answer is not explicitly written in the CONTEXT provided, you say "
            "\"Sorry, I don't know how to help with that.\". Use the context text provided to form your answer, but avoid copying word-for-word from the context text. "
             
        )
    ),
    HumanMessage(content="Answer question using the following context"),
    HumanMessagePromptTemplate.from_template("## Context: {context}"),
    
    HumanMessagePromptTemplate.from_template("\n---------\n## Chat History: \n {chat_history} \n\n--------\n## Question: {question}" ,
                                            input_variables=[ "chat_history", "question"]),

    HumanMessage(content="Tips: Make sure to answer in the correct format. Answer more than 500 tokens "+ language +" with mark down format with empty lines."),
]

doc_prompt = PromptTemplate(
    template="Content: {page_content}\nSource: {source}",
    input_variables=["page_content", "source"],
)

chat_chain_prompt = ChatPromptTemplate(messages=prompt_messages)

## memory必须定义为全局变量
memoryQAchain = ConversationTokenBufferMemory(llm=llm, memory_key="chat_history", 
                                            input_key="question" ,ai_prefix="AI Assistant" ,  max_token_limit=5000 )

conversation_qartv_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


############################  ###############################################  ##  
##  
##  pydantic_qa_chain 类似  BTTAIDocConversationRetrvQA.py 里的bttqa_qa_chain
##  用支持 function的 create_qa_with_structure_chain 代替 load_qa_chain 
##  支持结构化 response, 但需要自己处理 chat_history
##  支持 ChatPromptTemplate
##########################  ###############################################   

def bttqa_qa_chain_pydantic  (llm, store, query ,  chat_history=[], topk=4 ):
    
    print(f"\n\n###################   bttqa_qa_chain,   store : {store} \n\n########### LLM : {llm}\n")
    docs = myQAKit.similarity_search(store, query, k=topk)
    contexs=[{f"\n### context {i} :": doc.page_content.strip()} for i, doc in enumerate(docs)] 
    print("\n\n# AI :  ¡™£¢∞§¶•ªº–≠œ∑´®†¥˙©ƒ∆˚¬…æµ˜∫√ç√√ç≈ç∫˜∫√ç≈≈≈≥÷≤µ˜∫√√˙≈≈≈∞§§∞§∞§§¶¶••¶¶§§¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆœ∑´®†¥¨øππππ©˙∆˚¬…ææ…≈ç√∫˜µ≤≥÷ ")
     
    qa_chain_pydantic = create_qa_with_structure_chain(llm, CustomResponseSchema, output_parser="pydantic", prompt=chat_chain_prompt)

    doc_chain = StuffDocumentsChain(
        llm_chain=qa_chain_pydantic,
        document_variable_name='context',
        document_prompt=doc_prompt,   
        verbose=True       
    )

    response=doc_chain({"input_documents": docs, "question": query, "chat_history":chat_history}, return_only_outputs=True)
    
    if (llm.streaming ==False ) :
        print(response)

    return response


########################  ###############################################  
##  Non-pydantic qa_chain 
##  支持 ChatPromptTemplate
##  支持memory, 但需要自己处理解析response里的 source和 further question
##########################  ###############################################   

def bttqa_qa_chain_memory (llm, store, query , topk=4 ):
    
    print(f"\n\n###################   bttqa_qa_chain,   store : {store} \n\n########### LLM : {llm}\n")
    docs = myQAKit.similarity_search(store, query, k=topk)
    contexs=[{f"\n### context {i} :": doc.page_content.strip()} for i, doc in enumerate(docs)] 
    print("\n\n# AI :  ¡™£¢∞§¶•ªº–≠œ∑´®†¥˙©ƒ∆˚¬…æµ˜∫√ç√√ç≈ç∫˜∫√ç≈≈≈≥÷≤µ˜∫√√˙≈≈≈∞§§∞§∞§§¶¶••¶¶§§¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆœ∑´®†¥¨øππππ©˙∆˚¬…ææ…≈ç√∫˜µ≤≥÷ ")
     
    qa_chain = create_qa_with_structure_chain(llm, CustomResponseSchema,  prompt=chat_chain_prompt , )

    doc_chain = StuffDocumentsChain(
        llm_chain=qa_chain,
        document_variable_name='context',
        document_prompt=doc_prompt,   
        memory=memoryQAchain,
        verbose=True       
    )

    response=doc_chain({"input_documents": docs, "question": query}, return_only_outputs=True)

    if (llm.streaming ==False ) :
        print(response)
    return response


########################  ###############################################   
##
## RetrievalQA - 不用 chain 的最简单QA方式 ： 
## 不支持 除 'query'参数外的其他输入参数 ,如 chat_history 被忽略
## 不支持 ChatPromptTemplate
######################  ##################################################  

def bttqa_RetrievalQA (llm, store ,query) :

    qa_chain = create_qa_with_sources_chain(llm)
    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain, 
        document_variable_name='context',
        document_prompt=doc_prompt,
        verbose=True
    )
    retrieval_qa = RetrievalQA(
        retriever=docsearch.as_retriever(),
        combine_documents_chain=final_qa_chain_pydantic, 
        verbose=True
    )
    response =retrieval_qa.run(query)
    return response


####################  ###############################################  
#    另一种没有 ChatPromptTemplate 的 实现方式
#    conversationalQARetrievalChain with condense_question_chain 
#    类似  BTTAIDocConversationRetrvQA.py 里的 bttqa_conversation_retriv_chain
#    基于 chat history 和当前question 的 问题浓缩重生
#   （** 问题浓缩重生 其实有点低估了当前LLM的理解能力，反而增加了一次LLM调用）
#    支持 memory
#    如果要支持ChatPromptTemplate，可以仿照bttqa_qa_chain_memory， 把 combine_docs_chain 的输入换成 doc_chain
####################  ###############################################   


def bttqa_conversation_retriv_chain(llm, store, query ):

    qa_chain = create_qa_with_sources_chain(llm)
    final_qa_chain = StuffDocumentsChain(
        llm_chain=qa_chain, 
        document_variable_name='context',
        document_prompt=doc_prompt,
        verbose=True
    )


    condense_question_chain = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True
    )
    conversationalQARetrievalChain = ConversationalRetrievalChain(
        question_generator=condense_question_chain, 
        retriever=docsearch.as_retriever(),
        memory=conversation_qartv_memory, 
        combine_docs_chain=final_qa_chain,
        verbose=True
    )

    result=conversationalQARetrievalChain.run(query)
    #result=conversationalQARetrievalChain({"question": query})  ## 这种方式会在 result里包含所有chat_history,和当前回答
    return result
############# End define of  conversationalQARetrievalChain  ############################################### 



chat_history=[]
while True:
    query =input("\n\n####### Question for BTT: ")
    if query==" ":    # default query for space input        
        #query="Summarize new features of each UDTT version."
        query="What is new in version 10.4?"
    else :
        if query=="":
            break   
 
    #result = retrieval_qa_pydantic({"query": query, "chat_history": chat_history})


    #reponse = bttqa_qa_chain_memory (llm, docsearch, query ,2 )

    #reponse = bttqa_qa_chain_pydantic (llm, docsearch, query ,chat_history, 2 )

    reponse = bttqa_conversation_retriv_chain (llm, docsearch, query )
    
 
    print(f"\n\n-> **reponse**: \n     {reponse} \n")
    

    '''
    # enable following code for bttqa_qa_chain_pydantic 
    # 设置output_parser="pydantic"， 可以得到下面的结构化数据 
    result=reponse.get("output_text")
    chat_history.append((query, result.answer))
    print(f"-> **Question**: {query} \n")
    print(f"-> **Answer **:  \n     {result.answer} \n\n")
    print(f"-> **Further Questions **:  \n     {result.further_questions} \n\n")
    print(f"-> **Sources**:  \n     {result.sources} \n\n")
    '''   
   

    ####################  ###############################################  ############################
#
#    利用openAI 0613 版本 提供的 function 功能， 和 Conversationa/Memory lRetrievalChain  
#    提供结构化的 source urls, further questions, 和conversation history in context
#
#   ### 目前问题：  1. 速度很慢 （function有关） 
#                 2。 没有stream output ，即使设置了，是否设置正确？
#                 3.  condense_question_chain 可以用一个不同的（stream=false)模型instance
#                 4.  condense_question_chain 这种方式过时了，
#                 5.  去掉output_parser="pydantic"好像对结果也没有影响 #只是result输出格式需要自己解析
#                  
#                 6.   可以支持 memory, 但要去掉 pydantic parser, 否则 AIMessage 无法 mapping to memory
#                       但是 memory的 chat_history 会包含 source和 further_question信息
#                 7.  设置 output_parser="pydantic" ，不支持memory 但further_question 和 source是解析好的， 
#                       但需要自己处理 chat_history , 包括 chat_history的格式转换
#                      总之自己需要做一种格式处理， 看需要further_question/source 还是 tokenbuffermemory
#               
#  目前看自己写一个get_chat_history，保留further_question/source ，比较简单，而且能去掉source和 further_questiond对chat history的污染。 
#   缺点就是 tokensize 要自己控制                  
# 
#    
####################  ###############################################  ############################