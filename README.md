# RAG Assistant

Pipeline RAG (Retrieval-Augmented Generation) locale permettant d'indexer des documents PDF et de les interroger via un LLM local (Ollama).

## Fonctionnement

```
PDFs (data/) → découpage en chunks → embeddings (SentenceTransformer) → index FAISS
                                                                              ↓
Question utilisateur → embedding → recherche FAISS → chunks pertinents → LLM Ollama → réponse
```

1. Les PDFs placés dans `data/` sont chargés et découpés en chunks de 500 caractères.
2. Chaque chunk est transformé en vecteur via le modèle `all-MiniLM-L6-v2`.
3. Les vecteurs sont indexés dans FAISS pour une recherche rapide par similarité.
4. À chaque question, les 3 chunks les plus proches sont récupérés et injectés comme contexte dans le LLM.

## Prérequis

- Python 3.10+
- [Ollama](https://ollama.com/) installé et le modèle `llama3.1:8b` disponible

## Installation

```bash
# Cloner le dépôt
git clone <url-du-repo>
cd rag-assistant

# Créer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate

# Installer les dépendances
pip install -r requirements.txt

# Créer le dossier de données et y placer les PDFs
mkdir data
cp vos_documents.pdf data/
```

## Utilisation

### Recherche seule (sans LLM)

Lance une boucle interactive qui affiche les chunks retrouvés sans passer par le LLM :

```bash
python app.py
```

### RAG complet avec LLM local

Démarrer Ollama en arrière-plan, puis lancer le script :

```bash
# Démarrer Ollama et télécharger le modèle (une seule fois)
ollama serve
ollama pull llama3.1:8b

# Lancer le RAG complet
python ask_local_llm.py
```

## Structure du projet

```
rag-assistant/
├── data/               # Dossier des PDFs à indexer (non versionné)
├── app.py              # CLI interactive — recherche seule
├── ask_local_llm.py    # Intégration LLM via Ollama
├── rag_pipeline.py     # Pipeline RAG : chargement, chunking, embedding, index FAISS
├── requirements.txt    # Dépendances Python
└── .env                # Variables d'environnement (non versionné)
```

## Technologies utilisées

| Composant | Technologie |
|-----------|-------------|
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Index vectoriel | `faiss-cpu` |
| Extraction PDF | `pypdf` |
| LLM local | Ollama (`llama3.1:8b`) |

## Licence

Voir [LICENSE](LICENSE).
