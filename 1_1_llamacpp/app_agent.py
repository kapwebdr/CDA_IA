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

CACHE_DIR = "../cache_model"
DATA_DIR = "./data"
DB_DIR = "./vector_db"
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DB_DIR, exist_ok=True)

MODEL_CONFIGS = {
    "transformers": ModelConfig(
        model_type="transformers",
        model_path="unsloth/Llama-3.2-1B-Instruct"
    ),
    "llamacpp": ModelConfig(
        model_type="llamacpp",
        model_path="bartowski/Llama-3.2-1B-Instruct-GGUF"
    ),
    "mlx": ModelConfig(
        model_type="mlx",
        model_path="mlx-community/Llama-3.2-3B-Instruct-4bit"
    )
}

def setup_agent(model_type: str):
    """Configure l'agent avec le modèle choisi et les documents"""
    # 1. Chargement du modèle
    config = MODEL_CONFIGS[model_type]
    loader = ModelLoader()
    
    if model_type == "transformers":
        model, tokenizer = loader.load_transformers(config)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=2048
        )
        llm = HuggingFacePipeline(pipeline=pipe)
    elif model_type == "llamacpp":
        model, _ = loader.load_llamacpp(config)
        def pipe(text): return [{"generated_text": model(text, max_tokens=2048)['choices'][0]['text']}]
        llm = HuggingFacePipeline(pipeline=pipe)
    else:  # mlx
        from langchain_community.llms.mlx import MLX
        
        llm = MLX(
            model="mlx-community/Llama-3.2-3B-Instruct-4bit",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.95
        )

    # 3. Charger et préparer les documents
    documents = []
    
    # Chargement des fichiers du dossier data
    data_files = {
        'personnages': 'data/personnages_principaux.txt',
        'techniques': 'data/techniques_ninja.txt',
        'villages': 'data/villages_ninja.txt',
        'organisations': 'data/organisations.txt'
    }
    
    for category, filepath in data_files.items():
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Création d'un Document pour chaque section majeure
            sections = content.split('\n\n')  # Séparation par double saut de ligne
            for section in sections:
                if section.strip():  # Ignorer les sections vides
                    documents.append(Document(
                        page_content=section,
                        metadata={'category': category}
                    ))

    # Découpage des documents
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

    # 5. Création du prompt
    prompt = PromptTemplate.from_template("""
    Tu es un assistant expert sur l'univers de Naruto. Utilise le contexte fourni pour répondre aux questions.
    
    Contexte: {context}
    
    Question: {input}
    
    Réponse:""")

    # 6. Création des chaînes
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(
        vector_db.as_retriever(search_kwargs={"k": 3}),
        document_chain
    )

    return retrieval_chain

def main():
    print("\n=== Agent IA avec Base de Connaissances ===\n")
    
    # Sélection du modèle
    print("Modèles disponibles :")
    print("1. transformers")
    print("2. llamacpp")
    print("3. mlx")
    
    while True:
        choice = input("\nChoisissez votre modèle (1-3) : ")
        if choice in ['1', '2', '3']:
            model_types = {
                '1': 'transformers',
                '2': 'llamacpp',
                '3': 'mlx'
            }
            model_type = model_types[choice]
            break
        print("Choix invalide. Veuillez choisir un nombre entre 1 et 3.")

    # Configuration de l'agent
    print("Configuration de l'agent...")
    chain = setup_agent(model_type)
    
    print("\nAgent prêt ! Posez vos questions (ou 'quit' pour quitter)")
    
    while True:
        question = input("\nVous: ")
        if question.lower() == 'quit':
            print("\nAu revoir!")
            break
        
        try:
            response = chain.invoke({"input": question})
            print("\nAssistant:", response['answer'])
        except Exception as e:
            print(f"\nErreur: {str(e)}")

if __name__ == "__main__":
    main() 