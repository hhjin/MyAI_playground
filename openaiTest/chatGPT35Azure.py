
import os
import openai

openai.api_type = "azure"
openai.api_base = "https://bttaidoc.openai.azure.com/"
openai.api_version = "2024-02-01"
openai.api_key = os.environ.get('OPENAI_API_KEY_azure')

messages = []
system_msg = "You are a helpful assistant who always answer in Chinese."
messages.append({"role": "system", "content": system_msg})
message=""
while message != "quit()":
    message = input("\n\n You    : ")

    import time
    start_time = time.time()          
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        #engine="gpt4",
        engine="gpt-4o",
        messages=messages
    )
    replay = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": replay})
    print('\n\n#######\n GPT Azure: ' + replay )

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution Time: {execution_time:.2f} seconds")

    response=openai.Embedding.create(
        model="text-embedding-ada-002",
        engine="embedding2",
        input="The food was delicious and the waiter..."
    )
    
    print(" embedding usage:",response.usage.total_tokens)
    
    # print("embedding data:",response.data[0].embedding)
    

