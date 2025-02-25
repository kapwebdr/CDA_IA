import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
import torch

# 1. Charger le modèle et le tokenizer (utilisons un modèle plus petit)
MODEL_NAME = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float32
)

# 2. Créer un pipeline Transformers puis l'adapter pour LangChain
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=100
)

llm = HuggingFacePipeline(pipeline=pipe)

# 3. Charger et préparer les documents
documents = [
    "Naruto est un ninja de Konoha qui possède le démon renard à neuf queues.",
    "Zelda est un jeu d'aventure où Link doit sauver la princesse Zelda."
]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
docs = text_splitter.create_documents(documents)

# 4. Créer une base de données vectorielle FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vector_db = FAISS.from_documents(docs, embeddings)

# 5. Créer le prompt template
prompt = PromptTemplate.from_template("""
Contexte: {context}

Question: {input}

Réponse:""")

# 6. Créer la chaîne de documents
document_chain = create_stuff_documents_chain(llm, prompt)

# 7. Créer la chaîne de recherche
retrieval_chain = create_retrieval_chain(
    vector_db.as_retriever(),
    document_chain
)

# 8. Tests
questions = [
    "Qui est Naruto ?",
    "Quel est le pouvoir de Naruto ?"
]

for question in questions:
    print(f"\nQuestion: {question}")
    response = retrieval_chain.invoke({"input": question})
    print(f"Réponse: {response['answer']}")