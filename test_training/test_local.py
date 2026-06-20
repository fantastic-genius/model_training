import ollama

response = ollama.chat(model='llama3.2:3b', messages=[
  {'role': 'user', 'content': 'Describe the largest skyscraper in the world?'}
])
print(response['message']['content'])

