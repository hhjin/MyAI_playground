from typing import Any, Dict, List, Optional
from langchain import PromptTemplate ,LLMChain
import langchain
from langchain.chat_models import ChatOpenAI ,AzureChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import sys
import re

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage



# 英文700单词，800+token, 3800+字符(含空格）。 翻译成中文 1200+字符（字），1400+token, 如果假设英文字符1byte, 中文字2bytes. 则电子存贮中文信息密度高30%-50%（考虑空格：否-是）。印刷信息密度高50-100%
# max_words=700 & context=4K 对于普通文章是有余地的，但如包含大量代码或符号，则可能不够. GPT3.5 还是会偷懒，在这种场景下，把前面的润色了，后面的就copy.把 max_words 调小点会增加润色的比例
def split_text_into_chunks(text, max_words=430):  
    # Split the text by newline
    lines = text.split('\n')
    
    chunks = []
    current_chunk_words = 0
    current_chunk = ""
    #按行处理分段 《max_word
    for line in lines:
        words = line.split(' ')
        if current_chunk_words + len(words) > max_words:
            # 分段截止到前一行
            chunks.append(current_chunk)
            # 当前行新起一段
            current_chunk = line
            current_chunk_words = len(words)
            if  current_chunk_words > max_words : 
                # 遇到超长的一行
                #  split the current_chunk into sentences 句子拆分
                split_sentences = current_chunk.split('.')
                # 将句子合并成子段，直到子段的总词数达到max_words
               
                current_sub_chunk = ""
                current_word_count = 0

                for sentence in split_sentences:
                    sentence_words = sentence.split()
                    if current_word_count + len(sentence_words) <= max_words:
                        current_sub_chunk +=  sentence+". "
                        current_word_count += len(sentence_words)
                    else:
                        chunks.append(current_sub_chunk.strip())
                        current_sub_chunk = sentence
                        current_word_count = len(sentence_words)

                if current_sub_chunk:
                    chunks.append(current_sub_chunk.strip())  
                current_chunk = ""
                current_chunk_words = 0              

        else:
            # Add a newline character to keep the original format
            current_chunk += '\n' + line
            current_chunk_words += len(words)
    
    # Add the last chunk
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

#############     main start   #############################  
prompt_messages_1stTrans = [
    SystemMessage(
        content=(
            """你是一位精通简体中文的专业翻译，曾参与《纽约时报》和《经济学人》中文版的翻译工作，因此对于新闻和博客文章的翻译有深入的理解。我希望你能帮我将以下英文文章段落翻译成中文，风格与上述杂志的中文版相似。
规则：
- 翻译时要准确传达原文事实和背景。
- 保留特定的英文术语、数字或名字，并在其前后加上空格，例如："生成式 AI 产品"，"不超过 10 秒"。
- 保留复制原文的所有特殊符号，包括 空格，连续的换行'\n'，制表缩进'\t'等
 根据内容直译，不要遗漏任何信息"""
             
        )
    ),

    HumanMessagePromptTemplate.from_template("""
### 英文原文: 
{essay}

### 中文直译结果: 
""" ,input_variables=["essay"] )
]

prompt_messages_2ndtTrans = [
    SystemMessage(
        content=(
            """你是一位专业中文翻译，擅长对翻译结果进行二次修改和润色成通俗易懂的中文，我希望你能帮我将以下英文文章的中文翻译结果重新意译和润色。
规则：
- 这些博客文章包含机器学习或AI等专业知识相关，注意翻译时术语的准确性
- 保留特定的英文术语、数字或名字，并在其前后加上空格，例如："生成式 AI 产品"，"不超过 10 秒"。
- 保留复制原文的所有特殊符号，包括 空格，连续的换行'\n'，制表缩进'\t'等
- 基于直译结果重新意译和润色，意译和润色时务必对照原始英文，不要添加也不要遗漏内容，并以让翻译结果通俗易懂，符合中文表达习惯
"""
             
        )
    ),

    HumanMessagePromptTemplate.from_template("""
### 英文原文: 
{essay}

### 中文直译结果: 
{trans_1st}

### 请你基于直译结果重新意译和润色，意译和润色时务必对照英文原文，不要添加也不要遗漏内容，让翻译结果通俗易懂，符合中文表达习惯.
### 重新意译和润色后：
""" ,input_variables=["essay","trans_1st"] )
]

langchain.verbose = True
llmchat=AzureChatOpenAI(streaming=True,deployment_name="gpt35turbo-16k", max_tokens=1500, temperature=0, callbacks=[StreamingStdOutCallbackHandler()])
  
chat_prompt_1stTrans = ChatPromptTemplate(messages=prompt_messages_1stTrans)
chat_prompt_2ndTrans = ChatPromptTemplate(messages=prompt_messages_2ndtTrans)

chain1st = LLMChain(llm=llmchat, prompt=chat_prompt_1stTrans)
chain2nd = LLMChain(llm=llmchat, prompt=chat_prompt_2ndTrans)

fileName='HowToDoGreatWork.md'
output1stFileName=fileName.split('.')[0]+"_翻译.md"
output1stText=f"\n\n######################  {output1stFileName} ##########\n\n"
output2ndFileName=fileName.split('.')[0]+"_精译.md"
output2ndText=f"\n\n######################  {output2ndFileName} ##########\n\n"

output3rdFileName=fileName.split('.')[0]+"_翻译对照.md"
output3rdText=f"\n\n######################  {output3rdFileName} ##########\n\n"


with open(fileName, 'r', encoding='utf-8') as file:
    markdown_text = file.read()

chunks = split_text_into_chunks(markdown_text)
for txt in chunks:
    print(txt)
for i, chunk in enumerate(chunks):
    #if i!=4 :
     #   continue
    print(f"\n\n\n################################### chunk - {i}  \n")
    inputs1= {"essay": chunk}
    response1 = chain1st.run(inputs1)
    output1stText = output1stText + response1
    #print (f"\n 第一次翻译结果： " ,response1)

    inputs2= {"essay": chunk, "trans_1st":response1}
    response2=chain2nd.run(inputs2 )
    output2ndText = output2ndText + response2

    output3rdText=  "\n\n--------------------------------------------------------------------------------------\n"+chunk+"\n\n------------------------------------------------------------------------------------\n"+response1+"\n\n-----------------------------------------------------------------------------------\n"+response2

with open(output1stFileName, 'a', encoding='utf-8') as file1:
    file1.write(output1stText)

with open(output2ndFileName, 'a', encoding='utf-8') as file2:
    file2.write(output2ndText)

with open(output3rdFileName, 'a', encoding='utf-8') as file2:
    file2.write(output3rdText)
  


  