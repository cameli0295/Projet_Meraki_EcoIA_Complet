
# Projet IA – Optimisation de la consommation d’un réseau Meraki via l’IA
Étudiante : Cherchem Camélia
Fichier ZIP : Projet_Meraki_EcoIA_Complet.zip

---

## Contenu du dossier :

- `meraki_eco_ai_ready.py` : Script principal avec IA (Random Forest)
- `.env` : Fichier à remplir avec ta clé API Meraki si tu l’obtiens (facultatif)

---

## 🔧 Prérequis :

Installer les dépendances :
```
pip install numpy matplotlib scikit-learn python-dotenv
```

---

## ▶️ Lancer la démo (en simulation) :
```
python meraki_eco_ai_ready.py
```

Cela lancera :
1. Une collecte simulée de trafic réseau horaire
2. L’entraînement d’un modèle IA
3. L’affichage graphique de la prédiction des heures creuses

---

## 🔐 Activer la vraie API Meraki (facultatif) :
1. Crée ton compte Meraki (https://dashboard.meraki.com)
2. Va dans « Mon profil » → Active les API → Génére une clé
3. Remplis le fichier `.env` :
```
MERAKI_API_KEY=ta_clé_ici
```

> Quand tu auras une vraie clé API, tu pourras remplacer la collecte simulée dans le script par des appels réels (je peux t’aider).

---

💡 Le projet est prêt pour une démo orale, une extension vers une interface graphique, ou une connexion API réelle.
