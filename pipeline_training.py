import pandas as pd
import numpy as np
import os
import joblib
import chromadb
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#Configuration globale
DATA_PATH = "data/dataset.csv"  # Chemin vers votre CSV
MODEL_OUTPUT_DIR = "models"
CHROMA_DB_PATH = "chroma_db_data"
COLLECTION_NAME = "support_tickets_v1"
EMBEDDING_MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

def load_and_preprocess(filepath):
    """
    Étape 1 : Chargement et nettoyage des données textuelles.
    """
    print(f"[INFO] Chargement des données depuis {filepath}...")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Le fichier {filepath} est introuvable.")
    
    df = pd.read_csv(filepath)
    
    # 1. Gestion des valeurs nulles
    df['subject'] = df['subject'].fillna('')
    df['body'] = df['body'].fillna('')
    
    # 2. Fusion (Feature Engineering)
    df['text'] = df['subject'] + " " + df['body']
    
    # 3. Nettoyage basique (Lowercase + Espaces)
    # Note : On ne retire pas les stopwords ici car le modèle Transformer a besoin de contexte.
    df['text'] = df['text'].str.lower()
    df['text'] = df['text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    
    # 4. Suppression des lignes vides
    initial_len = len(df)
    df = df[df['text'] != '']
    print(f"[INFO] Nettoyage terminé. {initial_len - len(df)} lignes vides supprimées. Total: {len(df)} tickets.")
    
    return df

def generate_embeddings(df):
    """
    Étape 2 : Génération des vecteurs sémantiques (Embeddings).
    """
    print(f"[INFO] Chargement du modèle d'embedding : {EMBEDDING_MODEL_NAME}...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    print("[INFO] Génération des embeddings en cours...")
    # Encodage
    embeddings = model.encode(df['text'].tolist(), show_progress_bar=True, batch_size=32)
    
    # Normalisation L2 (Important pour la distance cosinus et la stabilité de la Logistic Regression)
    embeddings_normalized = normalize(embeddings, norm='l2')
    
    return embeddings_normalized, model

def store_in_chromadb(df, embeddings):
    """
    Étape 3 : Indexation dans la base vectorielle ChromaDB.
    """
    print(f"[INFO] Initialisation de ChromaDB dans '{CHROMA_DB_PATH}'...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    
    # Préparation des données
    ids = [str(i) for i in df.index]
    documents = df['text'].tolist()
    
    # Métadonnées (Gestion des NaN pour Chroma)
    metadatas = df[['type', 'language']].fillna('Unknown').to_dict(orient='records')
    
    # Insertion par batch pour éviter les timeouts mémoire
    batch_size = 1000
    total_docs = len(ids)
    
    print(f"[INFO] Démarrage de l'indexation de {total_docs} documents...")
    for i in range(0, total_docs, batch_size):
        end_i = min(i + batch_size, total_docs)
        collection.add(
            ids=ids[i:end_i],
            embeddings=embeddings[i:end_i].tolist(),
            metadatas=metadatas[i:end_i],
            documents=documents[i:end_i]
        )
    print(f"[INFO] Indexation terminée dans la collection '{COLLECTION_NAME}'.")

def train_classifier(X, y):
    """
    Étape 4 : Entraînement du modèle de classification (Logistic Regression).
    """
    print("[INFO] Préparation des données d'entraînement...")
    # Stratify assure la même distribution des classes dans train et test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[INFO] Entraînement du modèle sur {X_train.shape[0]} exemples...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    # Évaluation
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n[RESULTAT] Précision (Accuracy) : {acc:.2%}")
    print("-" * 30)
    print(classification_report(y_test, y_pred))
    print("-" * 30)
    
    return clf

def save_artifacts(model, class_labels):
    """
    Étape 5 : Sauvegarde du modèle pour l'API.
    """
    os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
    
    model_path = os.path.join(MODEL_OUTPUT_DIR, "classifier.joblib")
    classes_path = os.path.join(MODEL_OUTPUT_DIR, "classes.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(class_labels, classes_path)
    
    print(f"[INFO] Modèle sauvegardé : {model_path}")
    print(f"[INFO] Labels sauvegardés : {classes_path}")

def main():
    try:
        # 1. Pipeline de données
        df = load_and_preprocess(DATA_PATH)
        
        # 2. Pipeline d'Embeddings
        embeddings, _ = generate_embeddings(df)
        
        # 3. Pipeline ChromaDB (Stockage)
        store_in_chromadb(df, embeddings)
        
        # 4. Pipeline ML (Entraînement)
        clf = train_classifier(embeddings, df['type'])
        
        # 5. Sauvegarde
        save_artifacts(clf, clf.classes_)
        
        print("\n=== PIPELINE TERMINÉ AVEC SUCCÈS ===")
        
    except Exception as e:
        print(f"\n[ERREUR CRITIQUE] : {e}")

if __name__ == "__main__":
    main()