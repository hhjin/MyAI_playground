 
import os 
os.environ["AZURE_API_KEY"] = "b32e2759a73047b59f2298b902a030c7"
os.environ["AZURE_API_BASE"] = "https://bttaidoc2.openai.azure.com/"
os.environ["AZURE_API_VERSION"] = "2023-07-01-preview"


from litellm import completion
# azure call
response = completion(
    model = "azure/gpt35turbo-16k", 
    messages = [{ "content": "Hello, how are you?","role": "user"}]
)
print(response.choices[0])
 