from typing import Any, Dict, List, Optional
from langchain import PromptTemplate ,LLMChain
import sys
import time
sys.path.append('utils/')
from mylangchainutils import QA_Toolkit 
import langchain
from langchain.llms import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

###########################################################################
#  
# 最接近 next node JS 版本的复刻。 除了会话历史token控制还没加上以及会话历史的格式改进
#
############################################################################

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
  
system_prompt_template_postfix =""" If the CONTEXT includes source URLs you include them under a SOURCES heading before potential questions. The source URLs are indexed with regular express "/-\[\\d+\] : /".The index number is also put in the place of your answer text which is related with the source URL index. 
  Always include all of the relevant source urls from the CONTEXT, but never list a URL more than once (ignore trailing forward slashes when comparing for uniqueness). Never include URLs that are not in the CONTEXT sections. Never make up URLs."""

system_message_prompt = SystemMessagePromptTemplate.from_template(system_prompt_template)

example_human_template="""
 CONTEXT:
  Next.js is a React framework for creating production-ready web applications. It provides a variety of methods for fetching data, a built-in router, and a Next.js Compiler for transforming and minifying JavaScript code. It also includes a built-in Image Component and Automatic Image Optimization for resizing, optimizing, and serving images in modern formats.
 SOURCE: nextjs.org/docs/faq
  
 QUESTION: 
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

language="in Chinese"
#language=""  #default english

human_template = """CONTEXT:
    ${context}
-------
Chat History:
{chat_history}
-------
    USER QUESTION: 
    ${question} Answer {language} with mark down format with empty lines.""" #对于完成模型，最后的结束字符很重要，我不小心输入中文句号加空格代替. 结果不停的输出编造的url 

human_message_prompt = HumanMessagePromptTemplate.from_template(human_template ,input_variables=["context", "question", "language"] )

chat_prompt = ChatPromptTemplate.from_messages(
    ### with one shot in context learnning
    ### [system_message_prompt, example_human, example_ai, human_message_prompt]
    
     ### No in context learnning
    [system_message_prompt,  human_message_prompt]
)


def bttqa_chat_chain(llm, store, query , chat_prompt,  chat_history=[], topk=6 ):
    
    print(f"\n\n###################   bttqa_chat_chain,   store : {store} \n\n########### LLM : {llm}\n")
    docs = myQAKit.similarity_search(store, query, k=topk)
    
    contexs=[{f"\n### context {i} :": doc.page_content.strip()} for i, doc in enumerate(docs)] 
    inputs= {"context": contexs, "question":query.strip(), "language": language,"chat_history": chat_history}
    print("\n\nAI : ¡™£¢∞§¶•ªº–≠œ∑´®†¥˙©ƒ∆˚¬…æµ˜∫√ç√√ç≈ç∫˜∫√ç≈≈≈≥÷≤µ˜∫√√˙≈≈≈∞§§∞§∞§§¶¶••¶¶§§¡™£¢∞§¶•ªº–≠œ∑´®†¥¨ˆœ∑´®†¥¨øππππ©˙∆˚¬…ææ…≈ç√∫˜µ≤≥÷ ")
    
    qaChatchain = LLMChain(llm=llm, prompt=chat_prompt )
    response=qaChatchain.run(inputs)
    
    '''
    response= llm( #only support llmChat
        chat_prompt.format_prompt(
            context=contexs, question=query, language=language ,chat_history=chat_history
        ).to_messages()
    )  '''

    if (llm.streaming ==False ) :
        print(response)
    return response



#############     main start   #############################  
langchain.verbose = False
myQAKit=QA_Toolkit( )
storeOpenAI_Supabase=myQAKit.get_dbstore_supabase( table_name="udtt_ic1666c_langchain" , query_name="match_udtt_ic1666c_langchain")

llmAzureChat=myQAKit.get_chat_azure(streaming=True,deployment_name="gpt35turbo-16k", max_tokens=2300, temperature=0.23)

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

chat_history=[]
while True:
        query =input("\n\n####### Question for BTT: ")
        if query==" ":    # default query for space input        
          query="What is new in each version of UDTT?"
        else :
          if query=="":
                break
        
        result=bttqa_chat_chain(llmAzureChat, storeOpenAI_Supabase, query, chat_prompt ,chat_history ,8)               
        
        #chat_history.append((query, result))
