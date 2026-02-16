# Utilisation d'une image Python légère et officielle
FROM python:3.10

# Configuration des variables d'environnement pour éviter les fichiers .pyc et bufferiser les logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Définition du répertoire de travail dans le conteneur
WORKDIR /app

# 1. Installation des dépendances système (si nécessaire pour certaines lib ML)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copie et installation des requirements (Optimisation du cache Docker)
COPY requirements_k8s.txt .
RUN pip install --no-cache-dir -r requirements_k8s.txt

# 3. Création des dossiers pour les outputs (pour que les volumes K8s puissent s'y monter)
RUN mkdir -p models chroma_db_data

# 4. Copie du code et des données
# Note : En prod, les données viendraient d'un volume ou d'un bucket S3, pas du build.
COPY data/ ./data/
COPY pipeline_training.py .

# Commande par défaut (peut être surchargée par Kubernetes)
CMD ["python", "pipeline_training.py"]