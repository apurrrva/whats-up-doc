from langchain_community.llms import Ollama

llm = Ollama(model='mistral')

function = """
def sum():
    sum = 0
    for i in range(10):
        sum += i

    return sum
"""

prompt = f"""Write the following function's documentation in markdown format.

{function}"""

output = llm.invoke(prompt)

# for chunk in llm.stream(prompt):
#     print(chunk, end='')
#     output += chunk

print(f'\n\n\n{output}')
