# RATTAC
A Python implementation of the Retrieval-Automated Tool Techniques And Creation (RATTAC) framework.

### ⚠️ **IMPORTANT NOTICE** ⚠️  
**RATTAC is still under development. If you choose to run it locally, you do so at your own risk. C4AI is not responsible for any tools an LLM may generate or use through RATTAC, nor for any impact these tools may have on your system.  
For safer use, we strongly recommend running RATTAC in Google Colab or a sandboxed environment whenever possible.
Google Colab Link: https://colab.research.google.com/drive/1z1ID92kT-qBMP69xQno4AfW_2pbSDIQj?usp=sharing**

We at the Center for Applied AI have just made a brand new framework for Agentic AI! We call it Retrieval-Automated Tool Techniques And Creation (RATTAC). The RATTAC framework allows for any non-agentic AI model to become an agentic model with the power of RAG. That's right! The now seemingly arcane practice of RAG is coming back in full swing to help deliver Agentic AI that has never been seen before!


As the name implies, RATTAC's biggest advantage over most Agentic AI models out there is that it allows for autonomous and independent tool creation! That's right! AI Large Language Models (LLMs) are able to create their own tools now to help them best serve any prompt you throw at it! Want a file created? RATTAC allows for an AI LLM to do that! Want to read a file? The AI LLM can just make the tool! Want the time? Done. Tool created! Want the model to make you a Python iteration of Snake and load it at the same time? No sweat! RATTAC's versatile nature allows for any LLM (even those without tool capabilities) to become a super agent super easily!


RATTAC comes pre-packaged with a weather tool and a web search tool! There are instructions on how to add your own tools to RATTAC if you want an LLM to have access to a function/tool you have hand crafted! Otherwise, let RATTAC make all the tools itself!


### Package Requirements:
```
bs4
lxml_html_clean
newspaper3k
ollama
openai
pymilvus
PyPDF2
pyvirtualdisplay
regex
requests
selenium
sentence_transformers
webdriver-manager
```


### Other Requirements:
- For ChatGPT usage, a valid OpenAI key is required.
- For Ollama usage, a necessary GPU is required along with the model you wish to use.


### A Diagram of RATTAC's Inner Workings
![RATTAC](https://github.com/user-attachments/assets/17d26be5-84fb-45b0-b61e-45f820e23b62)
