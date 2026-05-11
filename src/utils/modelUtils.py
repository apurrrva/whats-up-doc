import inquirer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          SummarizationPipeline)

# Function to get the user's model choice
def getModelChoice():
    """
    Prompt the user to select a model from a list of available models.

    Returns:
    - str: The selected model.
    """
    models = ["llama3", "llama2", "codellama", "mistral", "phi3"]

    # Create an interactive list using inquirer
    questions = [
        inquirer.List(
            "model",
            message="Please select a model:",
            choices=models,
        ),
    ]

    # Store and return user's choice of model
    answer = inquirer.prompt(questions)

    return answer["model"]

# Custom prompt template for our LangChain
def returnTemplate():
    """
    Return a template string for generating documentation.

    Returns:
    - str: The template string.
    """
    template = """Given a script file in {language}, generate its documentation for each function. For each function in the script, document its name, arguments, return values, and a brief explanation of its logic.

Strictly use the following format for each function:

## Function Name: `function_name`

### Arguments
* `arg1` (type): Description of argument 1.
* `arg2` (type): Description of argument 2.
* ...

### Return Values
* `return_value1` (type): Description of return value 1.
* `return_value2` (type): Description of return value 2.
* ...

### Explanation of Function Logic:
1. Brief explanation of the function logic step by step.
2. ...
3. ...

Ensure that code within comments is not parsed and documented. Generate nothing else than what is asked.

{code}"""

    return template

# Setup LangChain with user model choice and custom prompt template
def setupLangChain(model):
    """
    Set up a language processing chain with the specified model.

    Args:
    - model (str): The language model to use.

    Returns:
    - PromptChain: The language processing chain.
    """
    # Initialize model with Ollama to inference using the Ollama local server
    llm = Ollama(model=model)

    template = returnTemplate()

    # Create prompt template with two variables for code and language
    prompt = PromptTemplate(
        template=template,
        input_variables=["code", "language"]
    )

    # Combine prompt template and llm model instance to get the LangChain
    chain = prompt | llm

    return chain

# Test function to use any pre-trained model on the huggingface-hub
def createPipeline(checkpoint, device):
    '''
    Create a transformers model summarization pipeline.

    Args:
    checkpoint - model checkpoint
    device (integer) - either 0 or 1, to specify if there exists a GPU
    '''
    pipeline = SummarizationPipeline(
        model=AutoModelForSeq2SeqLM.from_pretrained(checkpoint),
        tokenizer=AutoTokenizer.from_pretrained(
            checkpoint,
            skip_special_tokens=True,
            legacy=False
        ),
        max_new_tokens=1024,
        device=device
    )

    return pipeline
