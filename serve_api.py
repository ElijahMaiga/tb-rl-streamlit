# serve_api.py
# ------------------------------
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
from stable_baselines3 import DQN

app = FastAPI(title="TB Screening RL API")

# Load scaler and model saved from Colab
scaler = joblib.load("scaler.pkl")
model = DQN.load("tb_dqn_model.zip")  # ensure file in same folder

class Patient(BaseModel):
    data: list  # list of features in the same order as training

@app.get("/")
def root():
    return {"status":"ok", "service":"TB Screening RL API"}

@app.post("/predict")
def predict(p: Patient):
    x = np.array(p.data).reshape(1, -1)
    x = scaler.transform(x)
    action, _ = model.predict(x, deterministic=True)
    return {"action": int(action)}
