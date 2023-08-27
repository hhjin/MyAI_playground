
import openai
openai.api_key = os.getenv("OPENAI_API_KEY_openai")
completion=None;

print(openai.api_key )
for i in range(500):
  promptInput = input("\n Input your mesage for CHAT-GPT :")

    
  completion=openai.Completion.create(
          engine="gpt35turbo",
          model="text-davinci-003",
          prompt=promptInput ,
          temperature=0.98,
          max_tokens=350,
          top_p=1,
          frequency_penalty=0,
          presence_penalty=0,
          stop=None
        )
    
  print(completion.choices[0].text)

