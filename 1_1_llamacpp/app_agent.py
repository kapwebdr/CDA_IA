import os
from typing import List
from model_utils import ModelConfig, ModelLoader
from transformers import pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.chains import LLMChain

CACHE_DIR = "../cache_model"
DATA_DIR = "./data"
DB_DIR = "./vector_db"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

MODEL_CONFIG = ModelConfig(
    model_type="transformers",
    model_path="unsloth/Llama-3.2-1B-Instruct"
)

def setup_agent():
    """Configure l'agent avec la base de connaissances locale et la recherche web"""
    # 1. Chargement du modèle
    model, tokenizer = ModelLoader().load_transformers(MODEL_CONFIG)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=2048
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # 2. Configuration de la recherche web
    search = DuckDuckGoSearchRun()
    web_search_tool = Tool(
        name="Recherche Web",
        description="Recherche d'informations récentes sur internet",
        func=search.run
    )

    # 3. Chargement des documents locaux
    documents = []
    data_files = {
        'personnages': 'data/personnages_principaux.txt',
        'techniques': 'data/techniques_ninja.txt',
        'villages': 'data/villages_ninja.txt',
        'organisations': 'data/organisations.txt'
    }
    
    for category, filepath in data_files.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            sections = content.split('\n\n')
            for section in sections:
                if section.strip():
                    documents.append(Document(
                        page_content=section,
                        metadata={'category': category}
                    ))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=20,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    docs = text_splitter.split_documents(documents)

    # 4. Création de la base vectorielle
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        cache_folder=CACHE_DIR
    )
    vector_db = FAISS.from_documents(docs, embeddings)

    # 5. Création des prompts
    local_prompt = PromptTemplate.from_template("""
    Tu es un assistant expert sur l'univers de Naruto. Utilise le contexte fourni pour répondre aux questions.
    
    Contexte: {context}
    Question: {input}
    Réponse:""")

    web_prompt = PromptTemplate.from_template("""
    Tu es un assistant qui aide à trouver des informations sur Naruto en utilisant les résultats de recherche web.
    
    Résultats de recherche: {search_result}
    Question: {input}
    
    Synthétise une réponse basée sur les résultats de recherche:""")

    # 6. Création des chaînes
    local_chain = create_stuff_documents_chain(llm, local_prompt)
    retrieval_chain = create_retrieval_chain(
        vector_db.as_retriever(search_kwargs={"k": 3}),
        local_chain
    )
    
    web_chain = LLMChain(llm=llm, prompt=web_prompt)

    return retrieval_chain, web_chain, web_search_tool

def main():
    print("\n=== Agent IA avec Base de Connaissances et Recherche Web ===\n")
    
    # Configuration de l'agent
    print("Configuration de l'agent...")
    local_chain, web_chain, search_tool = setup_agent()
    
    print("\nAgent prêt ! Posez vos questions (ou 'quit' pour quitter)")
    print("Ajoutez '!web' à votre question pour forcer la recherche internet")
    
    while True:
        question = input("\nVous: ")
        if question.lower() == 'quit':
            print("\nAu revoir!")
            break
        
        try:
            if "!web" in question:
                # Recherche web
                question = question.replace("!web", "").strip()
                search_result = search_tool.run(f"Naruto {question}")
                response = web_chain.invoke({
                    "input": question,
                    "search_result": search_result
                })
                print("\nAssistant (Web):", response['text'])
            else:
                # Base de connaissances locale
                response = local_chain.invoke({"input": question})
                print("\nAssistant (Local):", response['answer'])
                
        except Exception as e:
            print(f"\nErreur: {str(e)}")

if __name__ == "__main__":
    main() 