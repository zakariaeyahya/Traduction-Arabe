from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
from traitement import find_most_similar_labels
import json

app = FastAPI()

# Liste pour stocker les données des requêtes
data_list = []

# Chargement des embeddings et des étiquettes
try:
    df_embeddings = pd.read_csv("combined_embeddings_all-mpnet-base-v2.csv")
    embeddings = df_embeddings.values
    labels = pd.read_excel('classeur1.ods', engine='odf')['preferredLabel'].values
except Exception as e:
    raise RuntimeError("Error loading embeddings or Excel file: " + str(e))

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Chargez le modèle et le tokenizer
model_name = "facebook/seamless-m4t-v2-large"
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
processor = AutoTokenizer.from_pretrained(model_name)

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def translate_text(text, src_lang, tgt_lang):
    try:
        logger.info(f"Tentative de traduction de '{text}' de {src_lang} vers {tgt_lang}")
        inputs = processor(text=text, src_lang=src_lang, return_tensors="pt")
        outputs = translation_model.generate(**inputs, tgt_lang=tgt_lang)
        
        logger.info(f"Sortie brute du modèle : {outputs}")
        
        if isinstance(outputs, tuple):
            logger.info("La sortie est un tuple")
            outputs = outputs[0]  # Prendre le premier élément si c'est un tuple
        
        if isinstance(outputs, torch.Tensor):
            logger.info("Conversion du tensor en liste")
            outputs = outputs.tolist()
        
        if not isinstance(outputs, list):
            logger.warning(f"Type de sortie inattendu : {type(outputs)}")
            return None
        
        if not outputs:
            logger.warning("La liste de sorties est vide")
            return None
        
        if isinstance(outputs[0], list):
            logger.info("La sortie est une liste de listes")
            decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        else:
            logger.info("La sortie est une liste simple")
            decoded = processor.decode(outputs, skip_special_tokens=True)
        
        logger.info(f"Texte décodé : {decoded}")
        
        if isinstance(decoded, list):
            return " ".join(decoded)
        return decoded
    
    except Exception as e:
        logger.error(f"Erreur lors de la traduction : {e}")
        return None

class TextRequest(BaseModel):
    text: str

@app.post("/predict/")
def predict(request: TextRequest):
    try:
        user_text = request.text
        logger.info(f"Texte reçu : {user_text}")
        
        # Translate Arabic to English
        translated_text = translate_text(user_text, src_lang="arb", tgt_lang="eng")
        if not translated_text:
            raise ValueError("La traduction a échoué")
        
        logger.info(f"Texte traduit : {translated_text}")
        
        # Get embeddings
        results = find_most_similar_labels(translated_text, embeddings, labels)
        
        # Ajouter les données à la liste
        data_list.append({
            "user_text": user_text,
            "translated_text": translated_text,
            "results": results
        })
        
        return {
            "user_text": user_text,
            "translated_text": translated_text,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Erreur dans la fonction predict : {e}")
        raise HTTPException(status_code=500, detail=str(e))
from fastapi.responses import FileResponse

@app.get("/download-data/")
def download_data():
    try:
        # Écrire les données dans un fichier JSON
        with open("data.json", "w") as json_file:
            json.dump(data_list, json_file, indent=4)
        
        return FileResponse("data.json", filename="data.json", media_type="application/json")
    
    except Exception as e:
        logger.error(f"Erreur lors du téléchargement des données : {e}")
        raise HTTPException(status_code=500, detail=str(e))

