# Projet: Traduction et Similarité de Textes en Darija et Anglais


 
## Description

Ce projet comprend trois scripts principaux : traitement.py, main.py, et app.py. Ensemble, ils forment un pipeline qui permet de traduire des textes en Darija vers l'anglais, de trouver des étiquettes similaires à partir d'un ensemble pré-embarqués, et d'interagir avec ces fonctionnalités via une API et une interface utilisateur Streamlit.



## Prérequis
 
Python 3.7 ou supérieur.
Installer les dépendances nécessaires :
pip install -r requirements.txt

pip install pandas scikit-learn sentence-transformers transformers torch fastapi pydantic streamlit requests

# Scripts
 
##  1. traitement.py:
Ce script gère le prétraitement du texte, le calcul des embeddings, et la traduction.

  Fonctionnalités :

Nettoyage de texte (clean_description).
Calcul des embeddings de phrases (get_embedding).
Traduction de texte (translate_text).
Recherche des étiquettes les plus similaires (find_most_similar_labels).
Chargement des embeddings et des étiquettes à partir des fichiers (load_embeddings_and_labels).


  Utilisation :
Le script charge les modèles nécessaires et définit des fonctions pour traiter et comparer les textes.


##  2. main.py
Ce script implémente une API FastAPI pour servir les fonctionnalités de traduction et de recherche de similarités.

Endpoints :

POST /predict/: Reçoit un texte en Darija, le traduit en anglais, et trouve les étiquettes les plus similaires.
GET /download-data/: Permet de télécharger les données des requêtes traitées sous forme de fichier JSON.
Utilisation :
Lancer l'application avec :
uvicorn main:app --reload

##  3. app.py
Ce script implémente une interface utilisateur avec Streamlit pour interagir avec l'API.

Fonctionnalités :

Entrer un texte en Darija.
Traduire le texte et afficher les étiquettes similaires.
Afficher les résultats de similarité et les étiquettes en anglais et en arabe.
Utilisation :
Lancer l'application Streamlit avec :
streamlit run app.py
##  4. embedding_creation.py
Ce script gère la création des embeddings pour les colonnes spécifiées d'un fichier Excel et les combine en utilisant des poids définis.

Fonctionnalités :

Nettoyage de la description (clean_description).
Chargement des données à partir d'un fichier Excel.
Création des embeddings pour les colonnes spécifiées en utilisant le modèle all-mpnet-base-v2.
Combinaison des embeddings avec des poids définis.
Sauvegarde des embeddings combinés et des données nettoyées dans un nouveau fichier Excel.
Utilisation :
Le script charge les modèles nécessaires et crée des embeddings pour les colonnes spécifiées d'un fichier Excel, puis les combine et les sauvegarde.
#  Instructions d'Installation et d'Exécution
1 Cloner le dépôt : 
git clone <URL_DU_DEPOT>
cd <NOM_DU_DEPOT>

2. Installer les dépendances :
pip install -r requirements.txt

3. Configurer les chemins de fichiers dans traitement.py :
Modifier combined_embeddings_file avec le chemin absolu vers combined_embeddings_all-mpnet-base-v2.csv.
Modifier excel_file_path avec le chemin absolu vers classeur1.ods.

4.Lancer l'API FastAPI :
uvicorn main:app --reload

5.Lancer l'application Streamlit :
streamlit run app.py

## Exemples d'Utilisation
Accéder à l'interface utilisateur Streamlit à l'adresse http://localhost:8501.
Utiliser l'API en envoyant des requêtes POST à http://localhost:8000/predict/ avec un texte en Darija dans le corps de la requête.
# Notes
Assurez-vous que les fichiers nécessaires (combined_embeddings_all-mpnet-base-v2.csv et classeur1.ods) sont disponibles et correctement référencés dans le script.
Vérifiez que les modèles nécessaires sont téléchargés et chargés correctement.
# Auteurs
Ce projet a été développé par YAHYA ZAKARIAE .

# Licence
Ce projet est sous licence MITLICENCE.
