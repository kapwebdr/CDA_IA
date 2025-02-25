# Utilisation d'une image Python de base légère
FROM python:3.12-slim

# Création du répertoire de travail
WORKDIR /app
COPY . /app/

# Installer les dépendances Python
RUN pip install --upgrade --no-cache-dir -r /app/requirements.txt

# Commande pour démarrer l'application Python
CMD ["python", "app.py"]