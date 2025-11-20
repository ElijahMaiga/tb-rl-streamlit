import streamlit as st
import joblib
import numpy as np
from stable_baselines3 import DQN

st.title("TB Screening RL Demo live")
st.write("Fill patient features")

# --------------------
# Load artifacts
# --------------------
scaler = joblib.load("scaler.pkl")
agent = DQN.load("tb_dqn_model.zip")   # <-- Correct RL model
st.write("Model loaded successfully.")

# MUST match the exact columns used during training
feature_cols = [
    'gender','temp0','hr0','rr0','spo20',
    'clinscore0','weight0','smoke','agesmoke',
    'fever0','wtloss0','cough0',
    'homefuel','workfumes','hiv','otherdiagnos'
]

# --------------------
# User inputs
# --------------------
inputs = []
for col in feature_cols:
    if col in ["gender", "smoke", "fever0", "wtloss0", "cough0", "hiv"]:
        value = st.number_input(col, value=0, step=1)
    else:
        value = st.number_input(col, value=0.0)
    inputs.append(value)

# --------------------
# Predict
# --------------------
if st.button("Recommend Action"):
    X = np.array(inputs).reshape(1, -1)
    X = scaler.transform(X)

    # Stable-Baselines predict
    action, _states = agent.predict(X)

    st.success(f"Recommended Action: **{action[0]}**")
