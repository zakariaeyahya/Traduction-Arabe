import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, SeamlessM4Tv2Model
from sentence_transformers import SentenceTransformer
import torch
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Load precomputed embeddings
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
embedding_model = SentenceTransformer('all-mpnet-base-v2')

class TextRequest(BaseModel):
    text: str

def get_embedding(text):
    cleaned_text = clean_description(text)
    embedding = embedding_model.encode([cleaned_text])
    return embedding

def clean_description(description):
    cleaned_text = description.lower()  # Convert to lowercase
    cleaned_text = re.sub(r'[^A-Za-z0-9]+', ' ', cleaned_text)  # Remove special characters
    return cleaned_text

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
        
        # Get embedding of translated text
        user_embedding = get_embedding(translated_text)
        
        # Calculate similarities
        similarities = cosine_similarity(user_embedding, embeddings)[0]
        indices = np.argsort(similarities)[::-1][:5]
        
        results = []
        for idx in indices:
            label_eng = labels[idx]
            label_arb = translate_text(label_eng, src_lang="eng", tgt_lang="arb") or "Traduction échouée"
            
            results.append({
                "label_eng": label_eng,
                "label_arb": label_arb,
                "similarity": float(similarities[idx])
            })
        
        return {
            "user_text": user_text,
            "translated_text": translated_text,
            "results": results
        }
    
    except Exception as e:
        logger.error(f"Erreur dans la fonction predict : {e}")
        raise HTTPException(status_code=500, detail=str(e))
@app.get("/test_translation/{text}")
def test_translation(text: str):
    eng_text = translate_text(text, src_lang="arb", tgt_lang="eng")
    arb_text = translate_text(eng_text, src_lang="eng", tgt_lang="arb") if eng_text else None
    return {"original": text, "to_english": eng_text, "back_to_arabic": arb_text}
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)