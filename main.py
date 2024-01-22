import numpy as np
from gensim.models import Word2Vec
from nltk.stem import PorterStemmer
from keras.models import Sequential
from tensorflow.keras.saving import load_model
from sklearn.preprocessing import StandardScaler

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

stemmer = PorterStemmer()
vector: Word2Vec = joblib.load("vector.pkl")
scaler: StandardScaler = joblib.load("scaler.pkl")
model: Sequential = load_model("model.keras")

def predict(prompt: str):
    prompt = [
        stemmer.stem(word)
        for word in prompt.split()
    ]
    prompt = vector.wv.get_mean_vector(prompt).tolist()
    prompt = np.asarray([prompt])
    prompt = scaler.transform(prompt)

    return float(model.predict(prompt)[0][0]) * 100

@app.get("/predict/{prompt}")
async def predict_text(prompt: str):
    try:
        return {
            'prompt': prompt,
            'prediction': predict(prompt)
        }

    except Exception as exception:
        raise HTTPException(status_code=500, detail=exception.__str__())
