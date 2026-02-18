#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def rag_system(kb, query, top_k=3):

    sentences = []
    categories = []

    # Flatten KB into sentence-level chunks
    for item in kb:
        category = item["category"]
        for sentence in item["content"]:
            sentences.append(sentence)
            categories.append(category)

    # Embedding using TF-IDF
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(sentences)

    # Transform the query
    query_vec = vectorizer.transform([query])

    # Compute similarity
    sims = cosine_similarity(query_vec, vectors).flatten()

    # Retrieve top-k sentences
    top_indices = sims.argsort()[-top_k:][::-1]
    retrieved_sentences = [sentences[i] for i in top_indices]
    retrieved_categories = [categories[i] for i in top_indices]

    # GENERATION: combine retrieved sentences into a final response
    context_text = " ".join(retrieved_sentences)
    generated_response = "Based on the retrieved documentation: " + context_text

    return {
        "query": query,
        "retrieved_context": retrieved_sentences,
        "categories": retrieved_categories,
        "response": generated_response
    }


# In[5]:


import os
import nbformat
from nbconvert import PythonExporter

ipynb_path = "/content/drive/MyDrive/warehouse_ai/RAG_module/RAG.ipynb"
py_path = "/content/drive/MyDrive/warehouse_ai/RAG_module/RAG.py"

with open(ipynb_path, 'r') as f:
    nb = nbformat.read(f, as_version=4)

exporter = PythonExporter()
script, _ = exporter.from_notebook_node(nb)

with open(py_path, 'w') as f:
    f.write(script)

print(f"Converted {ipynb_path} to {py_path}")

