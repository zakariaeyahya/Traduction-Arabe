import pandas as pd
import re
from sentence_transformers import SentenceTransformer

# Nettoyage 
def clean_description(description):
    cleaned_text = description.lower()  # Convertir en minuscules
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', cleaned_text)  # Supprimer les caractères spéciaux
    return cleaned_text

file_path = 'classeur1.ods'
df = pd.read_excel(file_path, engine='odf')
columns_to_keep = ['preferredLabel', 'altLabels', 'description']
df = df[columns_to_keep]

df['description'] = df['description'].apply(clean_description)
model = SentenceTransformer('all-mpnet-base-v2')

# création des embeddings et les enregistrer en CSV
def create_and_save_embeddings(column_name, model, df, model_name):
    embeddings = model.encode(df[column_name].astype(str))  # Créer les embeddings
    embeddings_df = pd.DataFrame(embeddings)  # Créer un DataFrame avec les embeddings
    embeddings_file_path = f'{column_name}_embeddings_{model_name}.csv'
    embeddings_df.to_csv(embeddings_file_path, index=False)  # Enregistrer les embeddings dans un fichier CSV
    print(f"Les embeddings pour la colonne {column_name} avec le modèle {model_name} ont été enregistrés dans {embeddings_file_path}")

create_and_save_embeddings('preferredLabel', model, df, 'all-mpnet-base-v2')
create_and_save_embeddings('altLabels', model, df, 'all-mpnet-base-v2')
create_and_save_embeddings('description', model, df, 'all-mpnet-base-v2')

# combinaison des embeddings pour chaque modèle
def combine_embeddings(model_name):
    preferredLabel_embeddings = pd.read_csv(f'preferredLabel_embeddings_{model_name}.csv')
    altLabels_embeddings = pd.read_csv(f'altLabels_embeddings_{model_name}.csv')
    description_embeddings = pd.read_csv(f'description_embeddings_{model_name}.csv')
    weights = {'preferredLabel': 0.5, 'altLabels': 0.3, 'description': 0.2}
    combined_embeddings = (preferredLabel_embeddings * weights['preferredLabel'] +
                           altLabels_embeddings * weights['altLabels'] +
                           description_embeddings * weights['description'])
    combined_embeddings.to_csv(f'combined_embeddings_{model_name}.csv', index=False)
    print(f"Les embeddings combinés pour le modèle {model_name} ont été enregistrés dans combined_embeddings_{model_name}.csv")

combine_embeddings('all-mpnet-base-v2')

output_file_path = 'nouveau_fichier.ods'
df.to_excel(output_file_path, index=False)
