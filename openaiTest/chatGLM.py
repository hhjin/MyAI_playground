
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY_openai")
openai.api_type="openai"
openai.api_key ="b2b709bf2a7c2fb8e7d5d0db142580a1.hS5AyGwYp9Q3okTu"
openai.api_base ="https://open.bigmodel.cn/api/paas/v4"

messages = []
system_msg = "You are a helpful assistant who always answer in Chinese."
messages.append({"role": "system", "content": system_msg})
message=""
while message != "quit()":
    message = input("\n You    : ")

    import time
    start_time = time.time()      
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        #model="gpt-3.5-turbo-16k",
        model="glm-4",
        messages=messages,
        max_tokens=100,

    )
    replay = response["choices"][0]["message"]["content"]
    messages.append({"role": "system", "content": replay})
    print(' GPT : ' + replay )
     
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")


    ''' # test Embedding
    response=openai.Embedding.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the waiter..."
    )
    print(" embedding usage:",response.usage.total_tokens)
    print("embedding data:",response.data[0].embedding)
    '''

