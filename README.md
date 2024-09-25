# A very CPU-friendly RAG implementation

A simple RAG tool for asking things related to a pdf file (in this project, it's about Don Quixote)

This project demonstrates how things can be implemented with CPU-friendly database and models. It uses
- [FAISS](https://github.com/facebookresearch/faiss) for vector database
- [WordLlama](https://github.com/dleemiller/WordLlama) for text embedding model
- **supposedly** [Local SmolLMs](https://huggingface.co/collections/HuggingFaceTB/local-smollms-66c0f3b2a15b4eed7fb198d0) for language model but I'm too lazy to setup on my Windows 10. Therefore, I use OpenAI API instead. I know it's possible!

## Setup

Python 3.10

Install required packages, please see `requirements.txt` for extra information
```bash
pip install -r requirements.txt
```

OpenAI API key is stored at `.env`

```bash
OPENAI_API_KEY=<key_here>
```

## Run

It is strongly advised to reach each .py file before running any command. 
By doing so, you get to understand the project more.

At root project, to setup the vector database
```bash
python setup_db.py
```

At root project, to query something then have a language model answer
```bash
python rag.py
```

## Great researcher/developer-friendly RAG frameworks

- [stanfordnlp/dspy](https://github.com/stanfordnlp/dspy)
- [neuml/txtai](https://github.com/neuml/txtai)
- [SylphAI-Inc/AdalFlow](https://github.com/SylphAI-Inc/AdalFlow)