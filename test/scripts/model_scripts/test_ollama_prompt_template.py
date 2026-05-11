from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate

llm = Ollama(model='codellama')

fn = """def sum():
    sum = 0
    for i in range(10):
        sum += i

    return sum
"""

template = """Write the documentation for this function in markdown:
{function}"""

prompt = PromptTemplate(input_variables=['function'], template=template)

chain = prompt | llm 

chain.invoke({'function': fn})
