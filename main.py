# IMPORT DEPENDENCIES
from bs4 import BeautifulSoup
from newspaper import Article
import ollama
from openai import OpenAI
import os
from pymilvus import MilvusClient
import regex as re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from sentence_transformers import SentenceTransformer
import shutil
import string
import subprocess
import sys
from webdriver_manager.firefox import GeckoDriverManager
from firefox import download_firefox, download_ublock, download_gecko

"""
# **Retrieval-Automated Tool Techniques And Creation** (RATTAC)
### **Author:** Devon Slonaker
### **Organization:** Center for Applied AI (C4AI)

### Adding in your own tools:
1. Make a file in `tool_functions` and `tool_descriptions` both with the same name of what you want your tool to be called.
2. In your tool file in `tool_functions`, populate it with a python function that is able to be called and ensure that the input is something a langauge model can easily understand. This means something like a string, number, an array, or something else easily understood by LLMs would suffice. See the `search` or `weather` tool functions for examples.
3. In your tool file in `tool_descriptions`, make sure that you have well documented information about what your tool does including what the input(s) should be and what it should output. It is also recommended to give a brief synopsis of your tool's overall function so an LLM reading the documentation can better understand the task. It is recommended to use an LLM to reword existing documentation or descriptions so the LLM reading it can better understand it in its own language. See the `search` or `weather` tool descriptions for examples.

### NOTE:
Any tool a model creates that you do not want later on must be deleted from `tool_functions` (and also preferably `tool_descriptions`) or it will repopulate upon a database refresh.
"""

# INITIALIZE VARIABLES FOR USE IN THE MAIN PROGRAM
# Model platform can be 'openai' or 'ollama'
model_platform = 'openai'
llm_model = "gpt-4o-mini"

# Options on what RATTAC can and can't do
allow_autonomous_pip_installs = True
allow_autonomous_chatgpt_prompts = True
print_verbose_outputs = True
remove_tool_upon_too_many_bad_calls = True

# ENTER IN YOUR OPEN-AI KEY HERE
KEY = ""

# DOWNLOAD FIREFOX FOR SELENIUM
if 'firefox' not in os.listdir():
    download_firefox()
if 'geckodriver' not in os.listdir():
    download_gecko()
if 'ublock.firefox.signed.xpi' not in os.listdir():
    download_ublock()
binary_loc = './firefox/firefox'
driver_loc = './geckodriver'
xpi_path = "ublock.firefox.signed.xpi"


# SET UP EMBEDDING MODEL FOR VECTOR DATABASE
def get_embedding(query_text: str):
    embedding_model = SentenceTransformer('intfloat/e5-small-v2')
    embeddings = embedding_model.encode(query_text, normalize_embeddings=True)
    return embeddings


# INITIALIZE/RE-INITIALIZE THE LANCEDB VECTOR DATABASE WITH PRE-GENERATED TOOLS
try:
    # Initialize Milvus Lite client
    db_path = "./agent_tools.db"
    milvus_client = MilvusClient(db_path)

    # Drop the existing collection if it exists
    collection_name = "tool_table"
    if collection_name in milvus_client.list_collections():
        milvus_client.drop_collection(collection_name)

    # Create collection with just the dimension
    dimension = len(get_embedding(''))
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
        index_params={
            # or "IVF_FLAT", "IVF_SQ8", etc.
            "index_type": "HNSW",
            # or "IP" (Inner Product), "COSINE"
            "metric_type": "L2",
            "params": {
                # Number of neighbor links for HNSW
                "M": 8,
                # Higher values give better index quality but slower build
                "efConstruction": 64
            }
        },
        auto_id=True)

    # Prepare data for insertion
    data = []
    for tool_name in os.listdir('tool_functions'):
        with open(f'tool_functions/{tool_name}', 'r', encoding='utf-8') as tool_file:
            code = tool_file.read()
        with open(f'tool_descriptions/{tool_name}', 'r', encoding='utf-8') as tool_file:
            description = tool_file.read()

        params = re.search(r'\((.*?)\)', code)
        params = params.group(1) if params else ''

        data.append({
            "tool": tool_name,
            "parameters": params,
            "code": code,
            "description": description,
            "vector": get_embedding(tool_name).tolist()
        })

    # Insert data into Milvus Lite
    milvus_client.insert(
        collection_name=collection_name,
        data=data)

    print("Data inserted into Milvus Lite successfully.")

except Exception as milvus_e:
    print("Milvus Lite database could not be created.")


# INITIALIZE QUERY FUNCTION
def db_search(query_string: str):
    return_data = {}

    # Perform a similarity search
    res = milvus_client.search(
    collection_name=collection_name,
    data=[get_embedding(query_string).tolist()],
    output_fields=["tool",
                   "parameters",
                   "code",
                   "description"],
    limit=1)[0]

    return_data['tool'] = res[0]['entity']['tool']
    return_data['parameters'] = res[0]['entity']['parameters']
    return_data['code'] = res[0]['entity']['code']
    return_data['description'] = res[0]['entity']['description']
    return_data['distance'] = 1 - res[0]['distance']
    return return_data

print(f"Test Query: {db_search('test')}")


# SET UP MODEL PLATFORM SWITCH
def call_model(user_prompt, system_prompt):
    if model_platform == 'openai':
        oai_client = OpenAI(api_key=KEY)
        response = oai_client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
            stream=False)

        # Extract the model's reply
        model_reply = response.choices[0].message.content
        return model_reply
    elif model_platform == 'ollama':
        response = ollama.generate(model=llm_model,
                                   prompt=user_prompt,
                                   system=system_prompt)
        model_reply = response['response']
        return model_reply
    else:
        raise ValueError("Invalid model platform")


# SET UP AUTONOMOUS PIP INSTALLS
# Set to True if you want the LLM to have the ability to install Pip packages.
# Set to False if you do not want the LLM to have the ability to install Pip packages.
autonomous_pip_prompt = (f"The function you create is allowed to call `install_python_packages(packages)`, which accepts one argument:\n"
                         f"- `packages`: A list of strings representing the names of Python packages to install. These packages may be dependencies required by the function you create.\n"
                         f"The `install_python_packages` function installs the specified packages and returns a string indicating whether the installation was successful or if an error occurred.\n"
                         f"For executing system-level commands, including package installation, use the `sys` and `subprocess` modules.")

if allow_autonomous_pip_installs:
    def install_python_packages(packages: list) -> str:
        try:
            for package in packages:
                # For normal installs
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            return "Success!"
        except Exception as pip_e:
            return f"Error: {pip_e}"


# SET UP AUTONOMOUS MODEL API CALLS
# Set to True if you want the LLM to have the ability to make calls to ChatGPT.
# Set to False if you do not want the LLM to have the ability to make calls to ChatGPT.

autonomous_model_prompt = (f"The function you create is allowed to call `use_model(user_prompt, system_prompt)`, which accepts two arguments:\n"
                             f"- `user_prompt` (required): A string containing the user's request.\n"
                             f"- `system_prompt` (optional): A string with predefined instructions for the model. By default, it is set to 'You are a friendly AI assistant.'\n"
                             f"The `use_model` function sends a prompt to a large language model (LLM) and returns its response as a string.")

if allow_autonomous_chatgpt_prompts:
    def use_model(user_prompt, system_prompt="You are a friendly AI assistant."):
        return call_model(system_prompt=system_prompt,
                          user_prompt=user_prompt)


# INITIALIZE FALLBACK MODEL RESPONSE
def model_converse(prompt):
    return call_model(system_prompt="You are a friendly AI assistant.",
                      user_prompt=prompt)


# INITIALIZE A CONTEXTUAL PROMPT FOR WHEN A TOOL RETURNS INFORMATION
def model_converse_with_context(prompt, model_context):
    return call_model(system_prompt="You are an agent who is an expert at answering user prompts with respect to context.",
                      user_prompt=(f"Below is some context to a user's prompt:\n"
                                   f"{model_context}\n"
                                   f"\n"
                                   f"Below is the user's actual prompt:\n"
                                   f"{prompt}\n"
                                   f"\n"
                                   f"Answer the user's prompt or summarize the prompt with respect to the context.\n"
                                   f"The context answers the user's prompt.\n"
                                   f"If something was stated to have failed, state that as the answer to the prompt, otherwise answer the prompt normally with respect to the context."))


# INITIALIZE A TOOL-USAGE PROMPT FOR AN LLM TO BE ABLE TO CALL A GIVEN TOOL
def use_tool(prompt, tool, tool_params, tool_description):
    if print_verbose_outputs:
        print("Awaiting Tool Usage Agent Response...")

    # The instruction is optimized to make universal, reusable tools for similar future queries
    return call_model(system_prompt="You are an agent who is an expert at using tools to complete a job.",
                    user_prompt=(f"### INSTRUCTION ###\n"
                                 f"You will be given a tool and description of what it does. Use the tool appropriately to fulfill the user's requirements.\n"
                                 f"Respond with JUST the tool function with the argument required to get the proper return value for the user's prompt.\n"
                                 f"The tool function with the argument you provide will be evaluated as Python code when you answer. It is imperative that you respond with JUST the tool and it's argument and nothing else.\n"
                                 f"\n"
                                 f"### TOOL ###\n"
                                 f"{tool}({tool_params}): {tool_description}\n"
                                 f"\n"
                                 f"### USER PROMPT ###\n"
                                 f"{prompt}\n"
                                 f"\n"
                                 f"Respond with just the tool and its argument with respect to the user's prompt."))


# INITIALIZE THE FUNCTION THAT IS TO BE CALLED FOR AN LLM TO MAKE ITSELF A NEW TOOL
def make_tool(prompt: str) -> dict:
    if print_verbose_outputs:
        print("Awaiting Tool Creation Agent Response...")

    # Call an LLM to make a Python function in code
    response = call_model(system_prompt="You are a Python programmer who is skilled at making function definitions.",
                          user_prompt=(f"Make a Python function that performs the requested task in a way that is reusable and adaptable to similar future requests. The function should not be hardcoded to specific values but should be written in a general manner that allows it to handle variations of the task dynamically. The function should be structured to accommodate any similar request by accepting parameters relevant to the task. The function must return a string indicating whether it was successful or if an error occurred.\n"
                                       f"Ensure your function name does not match any existing functions to avoid overriding them. The currently existing functions are: {[function_file for function_file in os.listdir('tool_functions') if not function_file.startswith('.')]}.\n"
                                       f"\n{autonomous_pip_prompt if allow_autonomous_pip_installs else ''}\n"
                                       f"\n{autonomous_model_prompt if allow_autonomous_chatgpt_prompts else ''}\n"
                                       f"Only provide Python code as raw text for the function without any additional explanation, formatting, or markdown. The code should not include calls to the function, only its definition.\n"
                                       f"Because you are making a Python function definition, the very first thing in your output should be 'def'.\n"
                                       f"Important: The function's return value must always provide a clear and contextual response. If the function executes successfully, the return string should dynamically reflect the nature of the output. The response format should be meaningful for another system (such as an LLM) that processes it. For example:\n"
                                       f"- A weather function should return 'Weather for {{location}}: {{weather_data}}'.\n"
                                       f"- A file-reading function should return 'Contents of {{filename}}: {{contents}}'.\n"
                                       f"- A data-processing function should return 'Processed data: {{result}}'.\n"
                                       f"- If the function does not naturally have a named output, return 'Success: {{result}}'.\n"
                                       f"The function must ensure that error messages are also explicit, such as 'Error: Invalid input' or 'Error: Unable to retrieve data'.\n\n"
                                       f"The specific user request is:\n\n"
                                       f"{prompt}\n\n"
                                       f"Now create a Python function definition according to the rules and user prompt above. Output your function definition as just raw text and do not use any additional formatting or provide any additional explanations."))

    function_reply = response

    # Call an LLM to create a description for the function if a valid one is returned
    if function_reply and re.search('^def ', function_reply):
        function = function_reply[4:].split('(')[0].strip()
        if function:
            model_description = call_model(system_prompt="You are a Python code reader who is an expert at documentation.",
                                  user_prompt=(f"Examine the function below and describe each parameter in simple terms, including the type of value it expects and its intended purpose. Then, provide a brief explanation of what the function does overall. Keep the response as plain text without using any special formatting like markdown, code blocks, or bullet points.\n\n"
                                               f"{function_reply}"))

            # Initialize variables for the new tool
            new_tool_name = function.replace('_', ' ')
            new_params = re.search(r'\((.*?)\)', function_reply)
            if not new_params:
                new_params = ''
            else:
                new_params = new_params.group(1)

            # Extract the description if it exists
            description_reply = model_description
            if description_reply:
                description_reply = description_reply.strip()
            else:
                description_reply = ''

            # Finalize the new tool
            new_content = {'tool': new_tool_name, 'parameters': new_params, 'code': function_reply, 'description': description_reply, 'vector': get_embedding(new_tool_name).tolist()}

            # Return the new tool
            return new_content
    return {}


# INITIALIZE A TOOL AUDIT TO DOUBLE-CHECK WITH THE MODEL IF IT THINKS THE CURRENT TOOL IS THE RIGHT TOOL FOR THE JOB
def check_tool(prompt, tool, tool_params, tool_description):
    if print_verbose_outputs:
        print("Awaiting Tool Audit Agent Response...")

    return call_model(system_prompt="You are an agent who is an expert at checking whether a tool is appropriate to complete a job.",
                      user_prompt=(f"### INSTRUCTION ###\n"
                                   f"You will be given a tool and description of what it does. Analyze the tool appropriately to check if it is the right tool to fulfill the user's requirements.\n"
                                   f"Respond with JUST \"True\" if the tool is appropriate for the job, or \"False\" if the tool is not appropriate for the job.\n"
                                   f"The response you provide will be evaluated as Python code when you answer. It is imperative that you respond with JUST \"True\" or \"False\" and nothing else.\n"
                                   f"\n"
                                   f"### TOOL ###\n"
                                   f"{tool}({tool_params}): {tool_description}\n"
                                   f"\n"
                                   f"### USER PROMPT ###\n"
                                   f"{prompt}\n"
                                   f"\n"
                                   f"Respond with just \"True\" if the tool is appropriate for the user's job, or \"False\" if it does not."))


# THE MAIN LOOP FOR TALKING TO RATTAC
print()
text = input("Ask RATTAC Agent: ")
while text != '/bye':
    # Perform a similarity search
    query_result = db_search(text)

    # Store results
    returned_tool = query_result['tool'].strip().replace(' ', '_')
    returned_parameters = query_result['parameters'].strip()
    returned_code = query_result['code'].strip()
    returned_description = query_result['description'].strip()
    returned_distance = query_result['distance']

    # See what LanceDB returns
    if print_verbose_outputs:
        print(f"RAG Tool Result: {returned_tool}")
        print(f"RAG Tool Distance: {returned_distance}")

    # Check if the current tool is the right tool for the job
    audit = check_tool(text, returned_tool, returned_parameters, returned_description).lower().translate(str.maketrans('', '', string.punctuation)).strip()
    audit_count = 1
    # Check to see if a new tool needs to be generated
    # Continually audit new tool until it is correct
    # Put a limit to save credits
    while audit != 'true' and audit_count <= 3:
        if print_verbose_outputs:
            print("Making New Tool...")

        # Make a new tool according to user's requirements
        new_tool = make_tool(text)
        if new_tool:
            returned_tool = new_tool['tool'].strip().strip().replace(' ', '_')
            returned_parameters = new_tool['parameters'].strip()
            returned_description = new_tool['description']
            returned_code = new_tool['code'].strip()

            # Audit new tool
            audit = check_tool(text, returned_tool, returned_parameters, returned_description).lower().translate(str.maketrans('', '', string.punctuation)).strip()

            # Skip if audit failed
            if audit != 'true':
                if print_verbose_outputs:
                    print(f"Wrong tool generated, trying again ({audit_count}/3)...")
                audit = 'false'
                audit_count += 1
                continue

            # If audit passed, save new tool
            milvus_client.insert(
                collection_name=collection_name,
                data=[new_tool])

            with open(f'tool_functions/{returned_tool}', 'w', encoding='utf-8') as file:
                file.write(returned_code)
            with open(f'tool_descriptions/{returned_tool}', 'w', encoding='utf-8') as file:
                file.write(returned_description)

            if print_verbose_outputs:
                print(f"Tool '{returned_tool}' successfully generated.")
        else:
            if print_verbose_outputs:
                print(f"Tool creation failed, trying again ({audit_count}/3)...")
            audit = 'false'
            audit_count += 1

    # Check audit to see if tool usage is allowed or if fallback to a standard LLM response is required
    if audit == 'true':
        if print_verbose_outputs:
            print(f"Tool Audit Successful!")

        # Get a model's tool usage response
        content = use_tool(text, returned_tool, returned_parameters, returned_description).strip()
        result = None
        usage_count = 1
        # Check to see if there is a valid result
        # Continually audit new tool until it is correct
        # Put a limit to save credits
        while not result and usage_count <= 3:
                try:
                    # Initialize the tool and get the result of the tool call
                    exec(returned_code)
                    result = eval(content)
                    if print_verbose_outputs:
                        print(f"Tool Usage Agent Response: {content}")
                except Exception as e:
                    if print_verbose_outputs:
                        print(f"Tool Usage Agent Response: {content}")
                        print(f"Tool Use Failed, trying again ({usage_count}/3)...")
                    result = None
                    usage_count += 1
                    # Get a model's tool usage response
                    content = use_tool(text, returned_tool, returned_parameters, returned_description).strip()

        if not result:
            if print_verbose_outputs:
                print(f"Tool Use Failed.")

            if remove_tool_upon_too_many_bad_calls:
                if print_verbose_outputs:
                    print(f"Removing Bad Tool...")
                try:
                    milvus_client.delete(
                        collection_name=collection_name,
                        where=f"tool = '{returned_tool.replace('_', ' ')}'")
                    os.remove(f'tool_functions/{returned_tool}')
                    os.remove(f'tool_descriptions/{returned_tool}')
                    print(f"Removed Bad Tool.")
                except Exception as e:
                    if print_verbose_outputs:
                        print(f"Tool Removal Failed.")
            result = ""
    else:
        if print_verbose_outputs:
            print(f"Tool Audit Failed.")
        result = ""

    # Summarize the outputs into a human-readable format
    model_final_response = ""
    if result != "":
        if print_verbose_outputs:
            print("Awaiting Summarizer With Context Agent Response...")
            print()
            model_final_response = model_converse_with_context(text, result).replace('\n\n', '\n')
        print(f"RATTAC Response: {model_final_response}")
    else:
        if print_verbose_outputs:
            print("Awaiting Summarizer Agent Response...")
            print()
            model_final_response = model_converse(text).replace('\n\n', '\n')
        print(f"RATTAC Response: {model_final_response}")
    print()
    text = input("Ask RATTAC Agent: ")
milvus_client.close()