# 10ms_RAGPipeline
This is a RAG Pipeline for the 10 Minute School Level One AI Engineer job. 

Hello, my name is Al Mahir Ahmed, this is my submission for the level 1 engineer job. Below I'll provide instructions and information related to the installation and the project itself. A small overview of the project beforehand, this pipeline uses an OCR based text extraction as it is more general and reliable. ChromaDB is used to store the vector embeddings, Ollama based models are primarily used as the LLM for this pipeline, and finally FastAPI is used as the lightweight conversational interface.

## Installation

There are quite a few things that are required to be installed before running the actual code since everything is done locally here.
```
pip install pytesseract pdf2image regex langchain-text-splitters langchain-chroma langchain-huggingface langchain-ollama sentence-transformers fastapi uvicorn
sudo apt install tesseract-ocr tesseract-ocr-ben
git clone https://github.com/Vulcan758/10ms_RAGPipeline.git
```
I apologize in advance if there are more packages that are required to be installed, I have not tested this on any other system apart from my own.

## Usage 

After installation, comes the usage. The main way to use this is by using the Swagger UI which comes from the FastAPI framework. Heres how you do that:

```
uvicorn api:app --reload
```

After doing this you now go to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), scroll to the ask pull down bar and edit the `text` value on the `Edit Value` portion. Then press execute and your request should go through and return a response. Keep in mind, due to the model running locally, it will take some time to generate a response but you will be able to monitor status updates on the command line as well as the health checker on the Swagger UI I asked one of the sample questions and the following as a response:

Question: "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"

Answer/Response: "এই কাজটি একটি বাক্য বা কবিতার জন্য সৃজনশীলভাবে অনুবাদ করা। এটি করার জন্য, আপনাকে প্রদত্ত পাঠটি বুঝতে হবে এবং একই অর্থ ধরে রাখতে এটিকে অন্য ভাষায় রূপ দিতে কয়েক শব্দে পরিবর্তন করতে হতে পারে।" 

I provided a more detailed guide explaining snippets of the code and my thought process in a tidy on the notebook file (.ipynb) named "Final RAG Notebook". There you will see how I got to constructing the main BanglaRAG class. If you go to the test directory and go into the .ipynb file there you will see a much messier version of my thought process along with a few things I tried testing before doing my final implementations.

### Some excuses and possible improvements

Due to myself having mid terms from the 29th of July I was not able to put as much time as I would've liked to on this. If I had more time, I would definitely implement a way to use both HuggingFace and Ollama LLMs at will instead of just Ollama LLMs. I would also provide an evaluation function to check how "correct" the pipeline/model performs. 

I had a lot of fun working on this, thank you 10 Minute School team!
