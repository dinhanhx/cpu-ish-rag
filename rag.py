import os
from pathlib import Path

import faiss
import pymupdf
from dotenv import load_dotenv
from openai import OpenAI
from wordllama import WordLlama

load_dotenv()

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

embedding_model = WordLlama.load(dim=1024)
embedding_index = faiss.read_index("embedding.index")

query = "Describe the Last Days of Don Quixote"
query_embedding = embedding_model.embed(query)
max_search_results = 1
source_distances, source_ids = embedding_index.search(
    query_embedding, max_search_results
)

file_path = Path("data/Don Quixote-www.learnenglishteam.com.pdf")
with pymupdf.open(file_path) as pdf:
    pdf.select(source_ids[0].tolist())
    # if max_search_results > 1, context will be the last page of the pdf
    # fix this before letting max_search_results > 1
    for page in pdf:
        context = page.get_text()

instruction = f"""Answer the following question based on the context below.

Question:
{query}

Context:
{context}
"""

chat_completion = openai_client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Answer the following question based on the context below.",
        },
        {"role": "user", "content": instruction},
    ],
)
answer = chat_completion.choices[0].message.content
print(answer)
