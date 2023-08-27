from typing import Any, Dict, List, Optional
from langchain import PromptTemplate ,LLMChain
from langchain.llms import Cohere
import sys
import time
import os
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 
import langchain
from langchain.chains.question_answering import load_qa_chain 
from langchain.chains import ConversationalRetrievalChain ,ConversationChain
from langchain.memory import ConversationBufferMemory ,ConversationTokenBufferMemory
 
from langchain.chains.conversational_retrieval.prompts import (
    CONDENSE_QUESTION_PROMPT,
    QA_PROMPT,
)

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

system_prompt_template = """You are a helpful assistant on IT area. When given CONTEXT you answer questions using only that information. You always format your output in markdown. After your answer, you act as an user instead of assistant by asking three specific insightful further questions 
  about the provided CONTEXT to identify the potential questions user want to know. You include code snippets if relevant. If you are unsure and the answer is not explicitly written in the CONTEXT provided, you say
  \"Sorry, I don't know how to help with that.\". Use the text provided to form your answer, but avoid copying word-for-word from the context text. 
  You try to use your own words when possible. If the CONTEXT includes source URLs you include them under a SOURCES heading before potential questions. The source URLs are indexed with regular express "/-\[\\d+\] : /".The index number is also put in the place of your answer text which is related with the source URL index. 
  Always include all of the relevant source urls from the CONTEXT, but never list a URL more than once (ignore trailing forward slashes when comparing for uniqueness). Never include URLs that are not in the CONTEXT sections. Never make up URLs."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)

example_human_template="""
 CONTEXT:
  Next.js is a React framework for creating production-ready web applications. It provides a variety of methods for fetching data, a built-in router, and a Next.js Compiler for transforming and minifying JavaScript code. It also includes a built-in Image Component and Automatic Image Optimization for resizing, optimizing, and serving images in modern formats.
 SOURCE: nextjs.org/docs/faq
  
 QUESTION: 
  what is nextjs?

---------------"""

example_human = HumanMessagePromptTemplate.from_template(example_human_template)

example_ai_template="""\n
 Next.js is a framework for building production-ready web applications using React. It offers various data fetching options, comes equipped with an integrated router, and features a Next.js compiler for transforming and minifying JavaScript. -[1] 
 Additionally, it has an inbuilt Image Component and Automatic Image Optimization that helps resize, optimize, and deliver images in modern formats.-[2]


 Sources:
  -[1]: https://nextjs.org/docs/faq
  -[2]: https://nextjs.org/docs/Image-Component

 Further Questions:
  - Q1: How does next.js built-in Image Component work ?
  - Q2: What is architecture of next.js ?
  - Q3: How to develop page flow with next.js?

  ---------------"""

example_ai = AIMessagePromptTemplate.from_template(example_ai_template)

language="in Chinese"
#language=""  #default english


human_template = """
    ## CONTEXT:
    ${context}

-------

    ## Chat History:
    {chat_history}

-------

    ## USER QUESTION: 
    ${question} Answer {language} with mark down format with empty lines.""" #对于完成模型，最后的结束字符很重要，我不小心输入中文句号加空格代替. 结果不停的输出编造的url 

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template ,input_variables=["context", "question", "language"] )

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, example_human, example_ai, human_message_prompt]
)


conversationMemoryQA_template = """You are a helpful AI Assistant on IT area. When given CONTEXT you answer questions using only that information. You always format your output in markdown. After your answer, you act as an user instead of assistant by asking three specific insightful further questions 
  about the provided CONTEXT to identify the potential questions user want to know. You include code snippets if relevant. If you are unsure and the answer is not explicitly written in the CONTEXT provided, you say
  \"Sorry, I don't know how to help with that.\". Use the text provided to form your answer, but avoid copying word-for-word from the context text. 
  You try to use your own words when possible. If the CONTEXT includes source URLs you include them under a SOURCES heading before potential questions. The source URLs are indexed with regular express "/-\[\\d+\] : /".The index number is also put in the place of your answer text which is related with the source URL index. 
  Always include all of the relevant source urls from the CONTEXT, but never list a URL more than once (ignore trailing forward slashes when comparing for uniqueness). Never include URLs that are not in the CONTEXT sections. Never make up URLs.
-----------
CONTEXT:
    ${context}

-----------

The following is a friendly conversation between a human and an AI Assistant. The AI Assistant is talkative and provides lots of specific details from aboved context. 
    
    Current conversation:
    
    {chat_history}
------------------
    Human: {question} You answer {language} with mark down format with empty lines.
    AI Assistant:"""


ConversationMemoryQA_PROMPT = PromptTemplate(input_variables=["context", "question",  "chat_history", "language"], template=conversationMemoryQA_template)
 
language="in Chinese"


memoryNoPrompt = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


#########################################################
#
# ####### 快来看传说中挽救了langchain 的  ==load_qa_chain==
# 支持 memory, 自定义参数的prompt ，自定义文档输入 和chain_type
#
#########################################################
def bttqa_qa_chain (llm, store, query , chat_prompt,  chat_history=[], topk=4 ):
    
    print(f"\n\n###################   bttqa_qa_chain,   store : {store} \n\n########### LLM : {llm}\n")
    docs = myQAKit.similarity_search(store, query, k=topk)
    contexs=[{f"\n### context {i} :": doc.page_content.strip()} for i, doc in enumerate(docs)] 
    print("\n\n# AI :  ¡™£¢∞§¶•ªº–≠œ∑´®†¥˙©ƒ∆˚¬…æµ˜∫√ç√√ç≈ç∫˜∫√ç≈≈≈≥÷≤µ˜∫√√˙≈≈≈∞§§∞§∞§§¶¶••¶¶§§¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆœ∑´®†¥¨øππππ©˙∆˚¬…ææ…≈ç√∫˜µ≤≥÷ ")
     
    chain = load_qa_chain(
        llm, chain_type="stuff",
        memory=memoryQAchain,
        prompt= chat_prompt 
        #prompt= ConversationMemoryQA_PROMPT
    )
    response=chain({"input_documents": docs, "question": query, "language":language}, return_only_outputs=True)

    if (llm.streaming ==False ) :
        print(response)
    return response



################################################################################
#
#  ConversationChain 虽然支持 propmt参数， 但 不支持 context 参数输入，写死了只支持 'history' 和  'input' : Error : The prompt expects ['history', 'question'], but got ['history'] as inputs from memory, and input as the normal input key.
#  
# - ## ConversationalRetrievalChain ： 
# ##  简单调用ConversationalRetrievalChain的方式， 但还是有question_generator的步骤，用的CONDENSE_QUESTION_PROMPT)
    #qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memoryNoPrompt ,verbose=True)
    #response = qa({"question": query})

    # 还有更定制化的调用，combine_docs_chain 可以支持自定义prompt, 但condense_question_chain是必须的（question_generator）
    # 不必处理context ，似乎从retrieverc传入 (用户控制不了 k)， chat_history 来自memory 
    # inputs= { "question":query.strip(), "language": language }  
    # response = conversationalQARetrievalChain(inputs)   
#   
#  总的来说， ConversationalRetrievalChain 不如 load_qa_chain 好用
#
################################################################################

 

def bttqa_conversation_retriv_chain(llm, store, query , chat_prompt,  chat_history=[], topk=1 ):
    
    print(f"\n\n###################   bttqa_conversation_retriv_chain,   store : {store} \n\n########### LLM : {llm}\n")

    retriever = store.as_retriever(lambda_val=0.025, k=1, filter=None)

    ''' retriever 的一个用法， 这里产生的context也没有实际用，打印看下内容而已
    docs = retriever.get_relevant_documents(query )    
    contexs=[{f"\n### context {i} :": doc.page_content.strip()} for i, doc in enumerate(docs)] 
    for  context in contexs :
        print(f"\n {context}")
    '''

    print("\n\nAI : ¡™£¢∞§¶•ªº–≠œ∑´®†¥˙©ƒ∆˚¬…æµ˜∫√ç√√ç≈ç∫˜∫√ç≈≈≈≥÷≤µ˜∫√√˙≈≈≈∞§§∞§∞§§¶¶••¶¶§§¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆœ∑´®†¥¨øππππ©˙∆˚¬…ææ…≈ç√∫˜µ≤≥÷ ")
    
       
    ##  简单调用ConversationalRetrievalChain的方式， 但还是有question_generator的步骤，用的CONDENSE_QUESTION_PROMPT)
    #qa = ConversationalRetrievalChain.from_llm(llm, retriever, memory=memoryNoPrompt ,verbose=True)
    #response = qa({"question": query})
   
    # 更定制化的调用，combine_docs_chain 可以支持自定义prompt, 但condense_question_chain是必须的（question_generator）， 
    condense_question_chain = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=chat_prompt)
    
    conversationalQARetrievalChain = ConversationalRetrievalChain(
        question_generator=condense_question_chain, 
        rephrase_question=True, # 不管 TrueFalse, condense_question_chain都会调用
        retriever=store.as_retriever(),
        combine_docs_chain=doc_chain,
        memory = conversationQARetriv_memory,
        verbose=True
    )

    inputs= { "question":query.strip(), "language": language }  # context 似乎从retrieverc传入， chat_history 来自memory
    response = conversationalQARetrievalChain(inputs)
     
    if (isinstance(llm, Cohere) or llm.streaming ==False ) :
      print("\n\n####### Answer : \n\t", response["answer"])
      print("\n\n####### Memory : \n\t", conversationQARetriv_memory.load_memory_variables({}) )

    return response


#############     main start   #############################  
langchain.verbose = True
myQAKit=QA_Toolkit("./Chroma_DB_300.2300")
storeOpenAI_Chroma=myQAKit.get_dbstore_openai()
#storeOpenAI_Supabase=myQAKit.get_dbstore_supabase()
store_Cohere=myQAKit.get_dbstore_cohere()
#store_SETF=myQAKit.get_dbstore_sentence()

llmAzureChat=myQAKit.get_chat_azure(streaming=True,deployment_name="gpt35turbo", max_tokens=1300, temperature=0,)
llmAzureChatNoneStreaming=myQAKit.get_llm_azure(streaming=False,deployment_name="gpt35turbo",  max_tokens=150, temperature=0, callbacks=None)

llmAzure300=myQAKit.get_llm_azure( streaming=True,max_tokens=300)
llmCohere=myQAKit.get_llm_cohere( max_tokens=200)  #cohere 不支持Chatchain ,&& Unicom Office network
 
 
conversationQARetriv_memory = ConversationTokenBufferMemory(llm=llmAzureChat, memory_key="chat_history",input_key="question", return_messages=True ,  max_token_limit=2000)
memoryQAchain = ConversationTokenBufferMemory(llm=llmAzureChat, memory_key="chat_history", input_key="question" ,ai_prefix="AI Assistant" ,  max_token_limit=5000 )
    
################################################################

chat_history=[]
while True:
        query =input("\n\n####### Question for BTT: ")
        if query==" ":    # default query for space input        
          query="What is UDTT used for?"
        else :
          if query=="":
                break
        
        response=bttqa_qa_chain (llmAzureChat, store_Cohere, query, chat_prompt,chat_history )     
        
        #response=bttqa_conversation_retriv_chain (llmAzureChat, store_Cohere, query, chat_prompt)  

        '''
        #answer=response.get("answer")
        answer=response 
        chat_history.append((query, response))

        print(f"-> **Question**: {query} \n")
        print(f"**Answer**: {answer} \n")          
        print(f"**response object**: {response} \n")     
        '''

####### 