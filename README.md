
# Projeto Grafo de Produtos com LightSAGE

Este projeto constrói um grafo de produtos utilizando PyTorch Geometric, treina um modelo LightSAGE para gerar embeddings dos produtos e disponibiliza uma API Flask para visualizar e recomendar produtos com imagens.

---

## Requisitos

- Python 3.12.10 ou superior
- GPU com CUDA - (recomendado para treino mais rápido)

---

## 1. Instalação

### Passo 1: Clone o repositório

```bash
git clone https://github.com/imxder/shopee-product-matching
cd shopee-product-matching
```

### Passo 2: Crie e ative o ambiente virtual

No Windows:

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

No Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Passo 3: Instale as dependências

```bash
pip install -r requirements.txt
```

## 2. Baixe as imagens.

- Execute o codigo de `baixar_imagens.py`.

```bash
python baixar_imagens.py
```

Este script deve:

- Baixar a pasta de imagens do Google Drive descomprimir.
- Descomprimir pasta `images.zip`.
- Salvar as imagens em: `static/images`

## 3. Gerar embeddings de texto e imagem.

Execute o script responsável por gerar os embeddings:

```bash
python gerar_embeddings.py
```

Este script deve:

- Criar o grafo de produtos com arestas baseadas em similaridades (ou regras definidas)
- Salvar os embeddings em `text_embeddings.npy` e `image_embeddings.npy`

## 4. Construir o grafo

Execute o script responsável pela construção do grafo. Por exemplo:

```bash
python construir_grafo.py
```

Este script deve:

- Carregar os dados do CSV.
- Criar o grafo de produtos com arestas baseadas em similaridades.
- Salvar o grafo em `produto_grafo.pt`.

---

## 5. Treinar o modelo LightSAGE

Execute o treinamento do modelo:

```bash
python train_lightSAGE.py
```

O que este script faz:

- Carrega o grafo salvo em `produto_grafo.pt`.
- Treina o modelo LightSAGE.
- Salva os embeddings resultantes em `gnn_embeddings.npy`.

> Ajuste parâmetros como batch size, número de épocas, e taxa de aprendizado no código conforme seu hardware.

---

## 6. Rodar a API Flask

Para iniciar a API Flask que exibe os produtos e recomendações:

```bash
python app.py
```

Acesse no navegador:

```
http://localhost:5000
```

Você verá a interface web com imagens dos produtos, recomendados pelo modelo.

---

## Estrutura dos arquivos
```
/
├── app.py                    # API Flask
├── build_graph.py            # Script para construir o grafo
├── train_lightSAGE.py        # Script para treinar o modelo LightSAGE
├── produto_grafo.pt          # Grafo salvo
├── gnn_embeddings.npy        # Embeddings de GNN salvos
├── text_embeddings.npy       # Embeddings de texto dos produtos
├── image_embeddings.npy      # Embeddings de imagem dos produtos
├── requirements.txt          # Dependências do projeto
├── train.csv                 # Dados dos produtos (CSV)
├── templates/                # Templates HTML do Flask
│   └── index.html
└── static/
    ├── css/
    │   └── style.css         # CSS 
    └── images/               # Imagens dos produtos 
```
