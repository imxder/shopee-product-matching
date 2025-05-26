
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
- Caso tenha placa de video e queira utilizar para treinar os modelos execute o código abaixo:
 ```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
-  Download de +/- 2GB!

## 2. Baixe as imagens.

- Execute o codigo de `baixar_imagens.py`.

```bash
python baixar_imagens.py
```

Este script deve:

- Baixar a pasta de imagens do Google Drive descomprimir.
- Descomprimir pasta `images.zip`.
- Salvar as imagens em: `static/images`

## 3. Baixe os modelos já treinados ou gere nos próximos passos.

- Execute o codigo de `baixar_modelos.py`.

```bash
python baixar_modelos.py
```

Este script deve:

- Baixar `models.zip` do Google Drive e descomprimir:
- `image_embeddings.npy`
- `text_embeddings.npy`
- `image_ids.npy`
- `produto_grafo.pt`
- `gnn_embeddings.npy`
- Salvar os modelos no diretório principal.

## 4. Gerar embeddings de texto e imagem.

Execute o script responsável por gerar os embeddings:

```bash
python gerar_embeddings.py
```

Este script deve:

- Criar o grafo de produtos com arestas baseadas em similaridades (ou regras definidas)
- Salvar os embeddings em `text_embeddings.npy` e `image_embeddings.npy`

## 5. Gerar o grafo

Execute o script responsável pela construção do grafo. Por exemplo:

```bash
python gerar_grafo.py
```

Este script deve:

- Carregar os dados do CSV.
- Criar o grafo de produtos com arestas baseadas em similaridades.
- Salvar o grafo em `produto_grafo.pt`.

---

## 6. Treinar o modelo GNN/lightSAGE

Execute o treinamento do modelo:

```bash
python train_GNN.py
```

O que este script faz:

- Carrega o grafo salvo em `produto_grafo.pt`.
- Treina o modelo GNN/LightSAGE.
- Salva os embeddings resultantes em `gnn_embeddings.npy`.

> Ajuste parâmetros como batch size, número de épocas, e taxa de aprendizado no código conforme seu hardware.

---

## 7. Rodar a API Flask

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
├── app.py                     # API Flask (Aplicação Principal)
├── gerar_embeddings.py        # Script para gerar embeddings iniciais
├── gerar_grafo.py             # Script para construir o grafo
├── train_GNN.py               # Script para treinar o modelo GNN
├── baixar_imagens.py          # Script para baixar imagens (via gdown)
├── baixar_modelos.py          # Script para baixar modelos .npy/.pt (via gdown)
|
├── train.csv                  # Dados dos produtos (CSV)
├── requirements.txt           # Dependências do projeto
|
├── image_embeddings.npy       # Embeddings de imagem (baixado ou gerado)
├── text_embeddings.npy        # Embeddings de texto (baixado ou gerado)
├── image_ids.npy              # IDs/Nomes das imagens (baixado ou gerado)
├── produto_grafo.pt           # Grafo salvo (baixado ou gerado)
├── gnn_embeddings.npy         # Embeddings GNN (baixado ou gerado)
|
├── templates/                 # Templates HTML do Flask
│   └── index.html             # Página principal da aplicação
|
├── static/                    # Arquivos estáticos (CSS, Imagens)
│   ├── style.css              # Arquivo CSS para estilização
│   └── images/                # Pasta para as imagens dos produtos (baixadas)
│       ├── image1.jpg
│       └── ...
|
└── uploads/                   # Pasta para imagens carregadas pelos usuários
    ├── upload1.jpg
    └── ...
```
