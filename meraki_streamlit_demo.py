
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import random

st.set_page_config(page_title="Meraki Eco IA", layout="wide")
st.title("🌿 Optimisation énergétique d’un réseau Meraki par IA")

st.markdown("## 1️⃣ Collecte des données simulées")
if st.button("Collecter les données réseau (simulation)"):
    traffic_data = [random.randint(5, 100) for _ in range(24)]
    st.session_state['traffic_data'] = traffic_data
    st.line_chart(traffic_data)
    st.success("Données simulées collectées")

if 'traffic_data' in st.session_state:
    st.markdown("## 2️⃣ Entraînement du modèle IA")
    if st.button("Entraîner le modèle IA"):
        X = np.array([[i] for i in range(24)])
        y = np.array(st.session_state['traffic_data'])
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        st.session_state['model'] = model
        st.success("Modèle IA entraîné avec succès")

    if 'model' in st.session_state:
        st.markdown("## 3️⃣ Prédiction des heures creuses & optimisation")
        if st.button("Prédire et visualiser les heures creuses"):
            X_pred = np.array([[i] for i in range(24)])
            predictions = st.session_state['model'].predict(X_pred)
            low_hours = [i for i, val in enumerate(predictions) if val < 30]
            st.success(f"Heures creuses détectées : {low_hours}")

            fig, ax = plt.subplots()
            ax.plot(range(24), predictions, marker='o', label="Prévision trafic")
            ax.axhline(y=30, color='red', linestyle='--', label="Seuil éco")
            ax.set_title("Prévision du trafic horaire")
            ax.set_xlabel("Heure")
            ax.set_ylabel("Trafic estimé")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)
