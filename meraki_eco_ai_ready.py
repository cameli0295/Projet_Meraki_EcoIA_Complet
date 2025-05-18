import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()
API_KEY = os.getenv("MERAKI_API_KEY")

def collect_data(simulation=True):
    print("[1/3] Collecte des données réseau...")
    time.sleep(2)
    if simulation or not API_KEY:
        # Simuler des données si aucune clé API
        traffic_data = [random.randint(5, 100) for _ in range(24)]
        print("Données simulées :")
    else:
        # À compléter plus tard : appel réel à l'API Meraki
        traffic_data = [random.randint(5, 100) for _ in range(24)]  # fallback temporaire
        print("Données réelles collectées (exemple simulé pour l’instant)")
    print(traffic_data)
    return traffic_data

def train_model(traffic_data):
    print("\n[2/3] Entraînement du modèle IA (Random Forest)...")
    time.sleep(2)
    X = np.array([[i] for i in range(24)])
    y = np.array(traffic_data)
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    print("Modèle entraîné avec succès")
    return model

def optimize_energy(model):
    print("\n[3/3] Prédiction des heures creuses...")
    predictions = model.predict(np.array([[i] for i in range(24)]))
    low_hours = [i for i, val in enumerate(predictions) if val < 30]
    print(f"Heures creuses identifiées : {low_hours}")

    # Affichage graphique
    plt.figure(figsize=(10, 4))
    plt.plot(range(24), predictions, marker='o', label="Prévision trafic")
    plt.axhline(y=30, color='red', linestyle='--', label="Seuil éco")
    plt.title("Prévision horaire du trafic réseau")
    plt.xlabel("Heure")
    plt.ylabel("Trafic estimé")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Exécution principale
if __name__ == "__main__":
    print("[INFO] Mode API Meraki activé :", "Oui" if API_KEY else "Non (simulation)")

    data = collect_data(simulation=not bool(API_KEY))
    model = train_model(data)
    optimize_energy(model)

