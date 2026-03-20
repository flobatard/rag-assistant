import subprocess
import sys
from pathlib import Path

PORT = 8501

if __name__ == "__main__":
    app_path = Path(__file__).parent / "streamlit_app.py"
    print(f"Démarrage de l'interface RAG sur http://localhost:{PORT}")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(PORT)],
        check=True,
    )
