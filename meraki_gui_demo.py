
import tkinter as tk
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import random
import time

class MerakiEcoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Meraki Eco IA - Démo")
        self.root.geometry("500x300")

        self.traffic_data = []

        self.label = tk.Label(root, text="Optimisation Meraki via IA", font=("Arial", 16))
        self.label.pack(pady=10)

        self.btn_collect = tk.Button(root, text="1. Collecter données", command=self.collect_data, width=30)
        self.btn_collect.pack(pady=5)

        self.btn_train = tk.Button(root, text="2. Entraîner IA", command=self.train_model, width=30)
        self.btn_train.pack(pady=5)

        self.btn_optimize = tk.Button(root, text="3. Optimiser consommation", command=self.optimize_energy, width=30)
        self.btn_optimize.pack(pady=5)

        self.status = tk.Label(root, text="Statut : prêt", fg="green")
        self.status.pack(pady=10)

        self.model = None

    def collect_data(self):
        self.status.config(text="Statut : collecte en cours...", fg="blue")
        self.root.update()
        time.sleep(1)
        self.traffic_data = [random.randint(5, 100) for _ in range(24)]
        self.status.config(text="✅ Données collectées (simulées)", fg="green")
        print("Données collectées :", self.traffic_data)

    def train_model(self):
        if not self.traffic_data:
            messagebox.showerror("Erreur", "Veuillez collecter les données d'abord.")
            return
        X = np.array([[i] for i in range(24)])
        y = np.array(self.traffic_data)
        self.model = RandomForestRegressor(n_estimators=10, random_state=42)
        self.model.fit(X, y)
        self.status.config(text="✅ Modèle entraîné avec succès", fg="green")

    def optimize_energy(self):
        if not self.model:
            messagebox.showerror("Erreur", "Veuillez entraîner le modèle d'abord.")
            return
        predictions = self.model.predict(np.array([[i] for i in range(24)]))
        low_hours = [i for i, val in enumerate(predictions) if val < 30]
        self.status.config(text=f"✅ Heures creuses détectées : {low_hours}", fg="green")

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

if __name__ == "__main__":
    root = tk.Tk()
    app = MerakiEcoApp(root)
    root.mainloop()
