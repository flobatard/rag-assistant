import requests

def ask_llm(context, question):
    prompt = f"""Tu es un assistant qui répond aux questions en te basant uniquement sur le contexte fourni.
Réponds en français, de manière claire et structurée. Si l'information n'est pas dans le contexte, dis-le explicitement.

Contexte:
{context}

Question: {question}

Réponse:"""

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.1:8b",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]