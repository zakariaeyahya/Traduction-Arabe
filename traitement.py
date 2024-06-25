import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

# Chemin absolu complet vers le fichier combined_embeddings_all-mpnet-base-v2.csv
combined_embeddings_file = r'D:\bureau\stage\exe 2\try\combined_embeddings_all-mpnet-base-v2.csv'

# Chemin absolu complet vers le fichier Excel
excel_file_path = r'd:\bureau\stage\exe 2\try\classeur1.ods'

# Chargement du modèle SentenceTransformer
model = SentenceTransformer('all-mpnet-base-v2')

# Chargement du modèle de traduction
model_name = "facebook/seamless-m4t-v2-large"
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
processor = AutoTokenizer.from_pretrained(model_name)

def clean_description(description):
    cleaned_text = description.lower()  # Convertir en minuscules
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', cleaned_text)  # Supprimer les caractères spéciaux
    return cleaned_text

def get_embedding(text):
    cleaned_text = clean_description(text)
    embedding = model.encode([cleaned_text])
    return embedding

def translate_text(text, src_lang, tgt_lang):
    try:
        inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
        outputs = translation_model.generate(**inputs, tgt_lang=tgt_lang)
        
        if isinstance(outputs, tuple):
            outputs = outputs[0]  # Prendre le premier élément si c'est un tuple
        
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.tolist()
        
        if not isinstance(outputs, list):
            return None
        
        if not outputs:
            return None
        
        if isinstance(outputs[0], list):
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        else:
            decoded = processor.decode(outputs, skip_special_tokens=True)
        
        if isinstance(decoded, list):
            return " ".join(decoded)
        return decoded
    
    except Exception as e:
        return None

def find_most_similar_labels(user_input, embeddings, labels, top_k=5):
    user_embedding = get_embedding(user_input)
    similarities = cosine_similarity(user_embedding, embeddings)[0]
    indices = similarities.argsort()[::-1][:top_k]
    
    results = []
    for idx in indices:
        label_eng = labels[idx]
        label_arb = translate_text(label_eng, src_lang="eng", tgt_lang="arb") or "Traduction échouée"
        
        results.append({
            "label_eng": label_eng,
            "label_arb": label_arb,
            "similarity": float(similarities[idx])
        })
    
    return results

# Chargement des embeddings et des étiquettes à partir des fichiers
def load_embeddings_and_labels():
    try:
        df_embeddings = pd.read_csv(combined_embeddings_file)
        embeddings = df_embeddings.values
        df_labels = pd.read_excel(excel_file_path, engine='odf')
        labels = df_labels['preferredLabel'].values
        return embeddings, labels
    except Exception as e:
        raise RuntimeError(f"Erreur lors du chargement des embeddings ou du fichier Excel : {str(e)}")
