import interpreter

interpreter.use_azure = True
interpreter.api_key = "de1c603ee9a84d3aa0c0b82ccbdde577"
interpreter.azure_api_base = "https://bttaidoc.openai.azure.com/"
interpreter.azure_api_version = "2023-07-01-preview"
interpreter.azure_deployment_name = "gpt35turbo-16k-func"

#interpreter.chat("Plot AAPL and META's normalized stock prices") # Executes a single command
interpreter.chat() # Starts an interactive chat