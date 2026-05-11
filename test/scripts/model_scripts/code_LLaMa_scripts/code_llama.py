from transformers import AutoTokenizer, pipeline
import torch
from utils import gpu_utils


def promptModel(prompt: str) -> None:
    """
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    """
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=512,
        batch_size=1
    )
    
    print("Chatbot:", sequences[0]['generated_text'])
    

# gpu_utils.checkGPU(tensorflow=False)
# gpu_utils.clearCUDACache()

model = "codellama/CodeLlama-7b-Instruct-hf" 

tokenizer = AutoTokenizer.from_pretrained(model, token=True)

# Note: RuntimeError: "addmm_impl_cpu_" not implemented for 'Half' error means that GPU implementation is only supported for float16 and not float32
# If your program runs on the CPU, turn this to float32
llama_pipeline = pipeline(
    "text-generation",  # LLM task
    model=model,
    torch_dtype=torch.float32,
    device_map="cpu"
)

testFunction = """
def get_llama_response(prompt: str) -> None:
    '''
    Generate a response from the Llama model.

    Parameters:
        prompt (str): The user's input/question for the model.

    Returns:
        None: Prints the model's response.
    '''
    sequences = llama_pipeline(
        prompt,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=256,
    )
    print("Chatbot:", sequences[0]['generated_text'])
"""

language = "Python"

prompt = f"""Here's my function in {language}:

{testFunction}

Given the definition of a function in any programming language (particularly Python and C++), please generate it's stand-alone documentation. I want it complete with fields like function name, function arguments and return values as well as a detailed explanation of how the function logic works line-by-line.  Make it concise and informative to put the documentation into a project documentation file."""

promptModel(prompt)
