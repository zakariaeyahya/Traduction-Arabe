import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re

# Chemin absolu complet vers le fichier combined_embeddings_all-mpnet-base-v2.csv
combined_embeddings_file = r'D:\bureau\stage\exe 2\try\combined_embeddings_all-mpnet-base-v2.csv'

# Chemin absolu complet vers le fichier Excel
excel_file_path = r'd:\bureau\stage\exe 2\try\classeur1.ods'

# Chargement du modèle SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

# Fonction pour nettoyer les descriptions
def clean_description(description):
    cleaned_text = description.lower()  # Convertir en minuscules
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', cleaned_text)  # Supprimer les caractères spéciaux
    return cleaned_text

# Fonction pour trouver les étiquettes les plus similaires
def find_most_similar_labels(user_input, combined_embeddings_df, df, top_k=5):
    user_embedding = model.encode([user_input])[0]  # Convertir l'entrée utilisateur en embedding
    similarities = cosine_similarity([user_embedding], combined_embeddings_df)[0]  # Calculer la similarité cosinus
    most_similar_indices = similarities.argsort()[::-1][:top_k]  # Trier et récupérer les indices des plus similaires
    top_k_labels = df.loc[most_similar_indices, 'preferredLabel'].values  # Récupérer les étiquettes correspondantes
    top_k_scores = similarities[most_similar_indices]  # Récupérer les scores de similarité
    return top_k_labels, top_k_scores

if __name__ == "__main__":
    try:
        # Chargement des embeddings combinés à partir du fichier CSV
        combined_embeddings_df = pd.read_csv(combined_embeddings_file)
        
        # Chargement du fichier Excel
        df_example = pd.read_excel(excel_file_path, engine='odf')
        
        # Exemple d'utilisation de la fonction find_most_similar_labels
        user_input_example = "recherche d'emploi"
        top_k_labels_example, top_k_scores_example = find_most_similar_labels(user_input_example, combined_embeddings_df.values, df_example)
        print(f"Exemple de texte utilisateur : {user_input_example}")
        for label, score in zip(top_k_labels_example, top_k_scores_example):
            print(f"Étiquette : {label}, Similarité : {score:.4f}")
    
    except FileNotFoundError:
        print(f"Le fichier {excel_file_path} n'a pas été trouvé.")
    except Exception as e:
        print(f"Une erreur s'est produite : {str(e)}")
