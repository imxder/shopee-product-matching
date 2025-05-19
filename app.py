from flask import Flask, render_template, request, send_from_directory
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# Carregamento dos dados
df = pd.read_csv("train.csv")
embeddings = np.load("gnn_embeddings.npy")

# Rota para servir imagens
@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory('static/images', filename)

# Página principal
@app.route("/", methods=["GET", "POST"])
def index():
    recomendados = []
    produto = None
    idx = None

    if request.method == "POST":
        try:
            idx = int(request.form.get("produto_idx"))
            if idx < 0 or idx >= len(df):
                raise ValueError("Índice fora do intervalo")
            produto = df.iloc[idx].to_dict()

            sim = cosine_similarity([embeddings[idx]], embeddings)[0]
            top_indices = sim.argsort()[::-1][1:6]
            recomendados = []
            for i in top_indices:
                rec = df.iloc[i].to_dict()
                rec["similaridade"] = sim[i]
                recomendados.append(rec)
        except Exception as e:
            return f"Erro: {e}"

    return render_template("index.html", produto=produto, recomendados=recomendados, max_index=len(df)-1)
    
if __name__ == "__main__":
    app.run(debug=True)
