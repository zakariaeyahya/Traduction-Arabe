from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import torch
from traitement import translate_text, find_most_similar_labels, load_embeddings_and_labels
import json
from fastapi.responses import FileResponse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import logging
from typing import List

app = FastAPI()

# Liste pour stocker les données des requêtes
data_list = []

# Chargement des embeddings et des étiquettes à partir des fichiers
try:
    embeddings, labels = load_embeddings_and_labels()
except Exception as e:
    raise RuntimeError("Error loading embeddings or Excel file: " + str(e))

# Chargement du modèle de traduction et du tokenizer
model_name = "facebook/seamless-m4t-v2-large"
translation_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
processor = AutoTokenizer.from_pretrained(model_name)

# Configuration de la journalisation
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
