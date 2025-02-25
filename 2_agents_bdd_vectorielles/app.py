import os
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import torch

# 1. Charger le modèle et le tokenizer
MODEL_NAME = "unsloth/Llama-3.2-1B"
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

# 1. Charger des données geeks (ex : descriptions d'animes et de jeux)
documents = ["Naruto est un ninja...", "Zelda est un jeu d'aventure..."]
text_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=20)
docs = text_splitter.create_documents(documents)

# 2. Créer une base de données vectorielle FAISS
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = FAISS.from_documents(docs, embeddings)

# 4. Créer un agent qui cherche dans la base vectorielle et répond avec le LLM
qa_chain = RetrievalQA(llm=llm, retriever=vector_db.as_retriever())
question = "Qui est Naruto ?"
answer = qa_chain.run(question)
print(answer)
question = "Quel est le pouvoir de Naruto ?"
answer = qa_chain.run(question)
print(answer)