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
#中文文本分段， word_count 指的是字数
def split_text_into_chunks(text, max_words=740):  
   
    #  split the current_chunk into sentences 句子拆分
    split_sentences = text.split('。')
    chunks=[]
    current_chunk_text = ""
    current_word_count = 0

    for sentence in split_sentences:
        sentence_words = sentence
        if current_word_count + len(sentence_words) <= max_words:
            current_chunk_text +=  sentence+"。 "
            current_word_count += len(sentence_words)
        else:
            chunks.append(current_chunk_text.strip())
            current_chunk_text = sentence
            current_word_count = len(sentence_words)

    if current_chunk_text:
        chunks.append(current_chunk_text.strip())   
    return chunks
 


prompt_template = """你是一位专业中文编辑，擅长对投稿文章进行二次修改和润色(Polish)成通俗易懂的中文，我希望你能帮我将以下文章润色。这些博客文章包含机器学习或AI等专业知识相关，注意时术语的准确性
- 保留原文中的英文单词和缩写,不要翻译成中文
- 保留特定的英文术语、数字或名字，并在其前后加上空格，例如："生成式 AI 产品"，"不超过 10 秒"。
- 保留复制原文的所有特殊符号
- 润色成通俗易懂的中文和符合中文表达顺序的语句调整，不要添加也不要遗漏内容，并以让结果通俗易懂，符合中文表达习惯
### 原文：
{essay}

### 用符合汉语表达习惯的语言润色文章(Polish), 请你避免直接复制原文。"""

prompt_messages_polish = [
    SystemMessage(
        content=(
            """你是一位专业中文编辑，擅长对投稿文章进行二次修改和润色(Polish)成通俗易懂的中文，我希望你能帮我将以下文章润色。这些博客文章包含机器学习或AI等专业知识相关，注意时术语的准确性
"""   )
    ),
    HumanMessagePromptTemplate.from_template("""
### 原文: 
{essay}


### 请你用符合汉语表达习惯的语言润色文章(Polish)
Rule:
- 保留原文中的英文单词和缩写,不要翻译成中文
- 保留特定的英文术语、数字或名字，并在其前后加上空格
- 保留原文的特殊符号,如[]等符号
""" ,input_variables=["essay","trans_1st"] )
]


essay="""
GPT4 或其他 LLMs 需要继续改进的方向包括：

- 信心校准：模型很难知道什么时候它应该有信心，什么时候它只是在猜测。模型会编造事实，我们称之为幻觉。如果是编造训练集里没有的内容属于开放域幻觉，如果是编造和prompt不一致的内容属于封闭域幻觉。幻觉可以用一种自信的、有说服力的方式陈述，所以很难被发现。有几种互补的方法来尝试解决幻觉问题。一种方法是改善模型的校准（通过提示或微调），使其在不可能正确的情况下放弃回答，或者提供一些其他可以用于下游的信心指标。另一种适合于缓解开放域幻觉的方法是将模型缺乏的信息插入到提示中，例如通过允许模型调用外部信息源，如搜索引擎（或其他 plugins）。对于封闭领域的幻觉，通过让模型对前文进行一致性检查会有一定程度的改善。最后，构建应用程序的用户体验时充分考虑到幻觉的可能性也是一种有效的缓解策略。
- 长期记忆：目前只有8000token（最新版可扩展到32k）。它以“无状态”的方式运行，且我们没有明显的办法来向模型教授新的事实。[1]
- 持续性学习：模型缺乏自我更新或适应变化环境的能力。一旦训练好，就是固定的。可以进行微调，但是会导致性能下降或过度拟合。所以涉及到训练结束后出现的事件、信息和知识，系统往往会过时。
- 个性化：例如，在教育环境中，人们期望系统能够理解特定的学习风格，并随着时间的推移适应学生的理解力和能力的进步。该模型没有任何办法将这种个性化的信息纳入其反应中，只能通过使用 meta prompts，这既有限又低效。
- 提前规划和概念性跳跃：执行需要提前规划的任务或需要Eureka idea的任务时遇到了困难。换句话说，该模型在那些需要概念性跳跃的任务上表现不佳，而这种概念性跳跃往往是人类天才的典型。[2]
- 透明度、可解释性和一致性：模型不仅会产生幻觉、编造事实和产生不一致的内容，而且似乎没有办法验证它产生的内容是否与训练数据一致，或者是否是自洽的。
- 认知谬误和非理性：该模型似乎表现出人类知识和推理的一些局限性，如认知偏差和非理性（如确认、锚定和基数忽略的偏差）和统计谬误。该模型可能继承了其训练数据中存在的一些偏见、成见或错误。
- 对输入的敏感性：该模型的反应对Prompts的框架或措辞的细节以及它们的顺序可能非常敏感。这种非稳健性表明，在Prompt 工程及其顺序方面往往需要大量的努力和实验，而在人们没有投入这种时间和努力的情况下使用，会导致次优和不一致的推论和结果。

**一些提高模型精准度的扩展手段：**

- 模型对组件和工具的外部调用，如计算器、数据库搜索或代码执行。
- 一个更丰富、更复杂的 "慢思考 "的深入机制，监督下一个词预测的 "快思考 "机制。这样的方法可以让模型进行长期的计划、探索或验证，并保持一个工作记忆或行动计划。慢思考机制将使用下一个词预测模型作为子程序，但它也可以获得外部的信息或反馈来源，并且它能够修改或纠正快速思考机制的输出。
- 将长期记忆作为架构的一个固有部分，也许在这个意义上，模型的输入和输出除了代表文本的标记外，还包括一个代表上下文的向量。
- 超越单个词预测：用分层结构代替标记序列，在嵌入中代表文本的更高层次的部分，如句子、段落或观点，内容是以自上而下的方式产生。目前还不清楚这种更高层次概念的顺序和相互依赖性的更丰富的预测是否会从大规模计算和“预测下一个词”的范式中涌现。

结语：**所以实际发生了什么？**

我们对GPT-4的研究完全是现象学的：我们专注于GPT-4能做的令人惊讶的事情，但我们并没有解决为什么以及如何实现如此卓越的智能的基本问题。它是如何推理、计划和创造的？**当它的核心只是简单的算法组件--梯度下降和大规模变换器与极其大量的数据的结合时，它为什么会表现出如此普遍和灵活的智能？**这些问题是LLM的神秘和魅力的一部分，它挑战了我们对学习和认知的理解，激发了我们的好奇心，并推动了更深入的研究。
"""

langchain.verbose = False
llmchat=AzureChatOpenAI(streaming=True,deployment_name="gpt35turbo", max_tokens=1500, temperature=0, callbacks=[StreamingStdOutCallbackHandler()])


'''
 ### hard code test
inputs= {"essay": essay}
PROMPT_test = PromptTemplate(   template=prompt_template, input_variables=["essay"]  )
chainTest = LLMChain(llm=llmchat, prompt=PROMPT_test)
chainTest.run(inputs)

chat_prompt = ChatPromptTemplate(messages=prompt_messages_polish)
chainPolish = LLMChain(llm=llmchat, prompt=chat_prompt)
chainPolish.run(inputs)
 ### end  test
'''

fileName='/Users/henryking/Desktop/AI/Doc/paper/什么是变压器模型及其工作原理？.txt'
output1stFileName=fileName.split('.')[0]+"_润色.md"
output1stText=f"\n\n######################  {output1stFileName} ##########\n\n"

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
    response1 = chainPolish.run(inputs1)
    output1stText = output1stText + response1
   

with open(output1stFileName, 'a', encoding='utf-8') as file1:
    file1.write(output1stText)
