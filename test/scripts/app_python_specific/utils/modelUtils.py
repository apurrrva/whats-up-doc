import inquirer
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          SummarizationPipeline)

def getModelChoice():
    models = ["llama3", "llama2", "codellama", "mistral", "phi3"]

    questions = [
        inquirer.List(
            "model",
            message="Please select a model:",
            choices=models,
        ),
    ]

    answer = inquirer.prompt(questions)

    return answer["model"]

def returnTemplate():
    template = """
    Given the definition of a function in Python, generate it's documentation. I want it complete with fields like function name, function arguments and return values as well as a detailed explanation of how the function logic works line-by-line. Make it concise and informative to put the documentation into a project.

Here is a sample function ```python
def log_directory_structure(directory_path, ai_context, indent=0):
    # Check if path is valid if not os.path.exists(directory_path):
        print("Directory not found.")
        return

    # Get the list of items in the directory items = os.alistair(directory_path)

    for item in items:
        item_path = os.path.join(directory_path, item)

        # Check if the item is a directory if os.path.sir(item_path):
            # Log the directory using ai_context directory_name = os.path.bename(item_path)
            indentation = "  " * indent  # Adjust indentation for subdirectories log_message = "Directory: ", directory_name"
            print(log_message)

            # Recursively log the subdirectory structure log_directory_structure(item_path, ai_context, indent + 1)

        # If it's a file, you can log it similarly

    # Get the directory path from the user directory_path = input("Enter the directory path: ")

    # Log the root directory print("Root Directory: ", directory_path")

    # Call the function to log the directory structure log_directory_structure(directory_path, ai_context)

    print(ai_context)
```

Here's an example of how to generate the documentation for the above function:

## Function Name: `log_directory_structure`

### Arguments
* `directory_path` (str): The path of the directory to be logged. It is required.
* `ai_context` (dict): A dictionary containing information about the AI context. This parameter can be used for additional functionality, but it's not necessary for logging the directory structure.

### Return Values None - this function does not return any value.

### Explanation of Function Logic:
1. The function first checks if the given path is valid by using `os.path.exists()`. If the path is invalid, it prints an error message and returns without logging anything.
2. It then gets a list of items in the directory using `os.alistair()` and iterates through each item.
3. For each item, it checks if it's a directory by using `os.path.sir()`. If it is, it logs the directory name using `print()`, adjusting the indentation level for subdirectories using string multiplication (`" " * indent`) and concatenating it with the current directory path.
4. It then recursively calls itself on the item path to log its subdirectory structure.
5. If the item is a file, it can be logged similarly by calling `print()` with the item name and path.
6. After logging all items in the directory, it prompts the user for the directory path using `input()` and logs the root directory using `print()`.
7. Finally, it calls itself on the given directory path to log its structure recursively.

Now, document the following function following how the example function's documentation looks like. This function is written in {language}. This is the function:

{function}"""
    
    return template

def setupLangChain(model):
    llm = Ollama(model=model)

    template = returnTemplate()

    prompt = PromptTemplate(
        template=template,
        input_variables=["function", "language"]
    )

    chain = prompt | llm

    return chain

def createPipeline(checkpoint, device):
    '''
    Create a transformers model summarization pipeline.

    Arguments:
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
