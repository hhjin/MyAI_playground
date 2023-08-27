
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY_openai")

messages = []
system_msg = input("What type of chatbot would you like to create?\n")
messages.append({"role": "system", "content": system_msg})
message=""
while message != "quit()":
    message = input("\n You    : ")
                
    messages.append({"role": "user", "content": message})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )
    replay = response["choices"][0]["message"]["content"]
    messages.append({"role": "system", "content": replay})
    print(' GPT3.5 : ' + replay )

 
    response=openai.Embedding.create(
        model="text-embedding-ada-002",
        input="The food was delicious and the waiter..."
    )
    
    print(" embedding usage:",response.usage.total_tokens)
 
    # print("embedding data:",response.data[0].embedding)
    

