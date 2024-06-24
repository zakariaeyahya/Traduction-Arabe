# Traduction Darija <-> Anglais avec Similarité

Ce projet permet de traduire du texte entre le darija (dialecte marocain) et l'anglais, de calculer des similarités et d'afficher les résultats dans une interface Streamlit.

## Fonctionnalités

- **Traduction Bidirectionnelle :** Traduction de textes entre le darija et l'anglais en utilisant le modèle SeamlessM4Tv2 de Hugging Face.
- **Calcul de Similarité :** Comparaison des embeddings de texte pour trouver les textes les plus similaires dans une base de données pré-enregistrée.
- **Interface Utilisateur :** Utilisation de Streamlit pour une interface utilisateur conviviale.

# Installation
 
## Installez les dépendances :
Assurez-vous d'avoir Python 3.7+ installé. Utilisez pip ou pipenv pour installer les bibliothèques nécessaires.

pip install -r requirements.txt
Préparez les fichiers nécessaires :

Assurez-vous d'avoir combined_embeddings_all-mpnet-base-v2.csv pour les embeddings combinés et classeur1.ods pour les étiquettes de votre base de données.
Utilisation
## Lancez le serveur FastAPI :
uvicorn main:app --reload
## Démarrez l'application Streamlit :
streamlit run app.py
Cela lancera l'interface utilisateur Streamlit sur http://localhost:8501.

## Utilisez l'interface utilisateur :

Entrez du texte en darija dans le champ prévu.
Cliquez sur le bouton pour traduire et afficher les résultats de similarité.
Structure des Fichiers
main.py : Contient le code FastAPI pour gérer les requêtes de traduction et de similarité.
app.py : Interface utilisateur Streamlit pour interagir avec l'API FastAPI.
combined_embeddings_all-mpnet-base-v2.csv : Fichier CSV contenant les embeddings combinés.
classeur1.ods : Fichier Excel avec les étiquettes pour la base de données.
Contribuer
Les contributions sont les bienvenues ! Si vous souhaitez améliorer ce projet, ouvrez une Pull Request ou signalez un problème dans les Issues.

Licence
Distribué sous la licence MIT. Voir LICENSE pour plus d'informations.

