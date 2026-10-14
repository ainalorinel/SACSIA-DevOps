# SACSIA — Système Intelligent de Contrôle et de Supervision des Accès

**SACSIA** est un projet de contrôle d'accès et de supervision intelligente du stationnement, combinant développement logiciel, vision par ordinateur et systèmes embarqués.

L'objectif est de concevoir une solution capable de contribuer à la sécurisation des accès et à la surveillance des places de parking grâce à plusieurs technologies complémentaires.

> Projet de Licence — Informatique, Électronique et Télécommunications.

## 🎯 Objectifs du projet

- Automatiser et améliorer le contrôle des accès.
- Exploiter la vision par ordinateur pour la reconnaissance et la détection.
- Superviser l'occupation des places de parking.
- Intégrer des systèmes embarqués pour interagir avec les équipements physiques.
- Centraliser les échanges entre le serveur et les composants du système.

## ✨ Fonctionnalités

Les fonctionnalités prévues ou implémentées dans le projet comprennent notamment :

- **Reconnaissance faciale :** utilisation de modèles de vision par ordinateur.
- **Détection des places de parking :** analyse des images pour identifier les zones de stationnement.
- **API backend :** serveur FastAPI pour gérer les échanges entre les composants.
- **Base de données :** MongoDB pour la persistance des informations.
- **Systèmes embarqués :** intégration prévue ou réalisée avec ESP32 et des périphériques de contrôle d'accès.
- **Modèles IA :** utilisation de modèles YOLO pour les tâches de vision par ordinateur.

La disponibilité de chaque fonctionnalité dépend de la configuration matérielle, logicielle et de la base de données.

## 🛠️ Technologies utilisées

| Technologie | Rôle |
|---|---|
| Python | Langage principal |
| FastAPI | API backend |
| MongoDB / PyMongo | Gestion des données |
| OpenCV | Traitement d'images |
| YOLO / Ultralytics | Détection par vision par ordinateur |
| face-recognition / dlib | Traitement lié à la reconnaissance faciale |
| ESP32 | Intégration des systèmes embarqués |
| Git & GitHub | Gestion du code source |
| GitHub Actions | Intégration continue (CI) |

## 📁 Structure du projet

```text
SACSIA_BASE/
├── .github/
│   └── workflows/
│       └── ci.yml
├── runs/
│   ├── facial_recognition/
│   │   └── train-6/weights/best.pt
│   └── detect/
│       └── model_for_parking/weights/best.pt
├── parking_spots.json
├── requirements.txt
├── server.py
├── setup_parking_spots.py
├── test.py
├── train.py
├── .gitignore
├── .python-version
└── README.md
```

Les dossiers de données locales, les environnements virtuels et les fichiers sensibles ne doivent pas être publiés dans le dépôt.

## ⚙️ Prérequis

- Python 3.13, conformément à la configuration actuelle du projet.
- Git.
- MongoDB local ou une instance MongoDB accessible à distance.
- Les fichiers de modèles IA nécessaires au serveur.
- Les dépendances Python listées dans `requirements.txt`.

Certaines dépendances, notamment celles liées à dlib et à la vision par ordinateur, peuvent nécessiter des prérequis système selon l'environnement.

## 🚀 Installation locale

### 1. Cloner le dépôt

```bash
git clone https://github.com/ainalorinel/SACSIA-DevOps.git
cd SACSIA-DevOps
```

### 2. Créer un environnement virtuel

Sous Windows PowerShell :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Installer les dépendances

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configurer MongoDB

Par défaut, la configuration locale utilise :

```text
mongodb://localhost:27017/
```

Pour utiliser une instance distante, configure la variable d'environnement `MONGODB_URI` si le serveur est configuré pour la lire.

**Sécurité :** ne publie jamais les identifiants MongoDB, les mots de passe ou les clés secrètes dans le dépôt Git.

### 5. Démarrer le serveur

```powershell
python server.py
```

Le serveur est configuré pour écouter sur le port `8000`.

Lorsque le serveur fonctionne, l'interface de documentation FastAPI est généralement accessible à l'adresse :

http://127.0.0.1:8000/docs

Cette adresse est locale à l'ordinateur qui exécute le serveur.

## 🔄 Intégration continue avec GitHub Actions

Le workflow situé dans `.github/workflows/ci.yml` vérifie actuellement la syntaxe des fichiers Python suivants :

- `server.py`
- `setup_parking_spots.py`
- `test.py`
- `train.py`

Le workflow se déclenche lors d'un push sur `main`, d'une pull request vers `main` ou manuellement depuis GitHub Actions.

Cette vérification constitue une première étape de CI. Elle ne garantit pas à elle seule que l'application démarre correctement, que MongoDB est accessible ou que tous les tests fonctionnels réussissent.

Consulter les exécutions : [GitHub Actions](https://github.com/ainalorinel/SACSIA-DevOps/actions).

## 📦 Modèles et données

Les modèles IA et le fichier `parking_spots.json` participent à la configuration du système.

Les modèles nécessaires au démarrage doivent être présents aux chemins attendus par le serveur. Les données personnelles, les images de personnes et les bases de données locales doivent être protégées et exclues du dépôt public.

## 🔐 Sécurité et confidentialité

Comme SACSIA peut traiter des données liées au contrôle d'accès et à la reconnaissance faciale :

- Ne jamais publier de photos de membres ou de données biométriques.
- Ne jamais versionner de mots de passe ou de chaînes de connexion contenant des identifiants.
- Utiliser des variables d'environnement pour les secrets.
- Limiter les accès à la base de données.
- Vérifier les autorisations réseau avant d'exposer le serveur sur Internet.
- Tester les mécanismes de contrôle d'accès dans un environnement maîtrisé avant toute utilisation réelle.

## 🗺️ Perspectives d'évolution

- Renforcer les tests automatisés.
- Ajouter des vérifications de dépendances dans le pipeline CI.
- Configurer le déploiement continu.
- Déployer l'API dans un environnement accessible en ligne.
- Configurer une base MongoDB distante de manière sécurisée.
- Améliorer la supervision, les journaux et la gestion des erreurs.

## 👨‍💻 Dépôt du projet

**GitHub :** [ainalorinel/SACSIA-DevOps](https://github.com/ainalorinel/SACSIA-DevOps)

---

*Documentation du projet SACSIA — contrôle d'accès et supervision intelligente du stationnement.*