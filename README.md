# Sistema de Diagnóstico de Diabetes

Este projeto utiliza Inteligência Artificial para auxiliar no diagnóstico de diabetes com base em dados clínicos.

## 📂 Organização dos Diretórios

A estrutura do projeto está organizada da seguinte forma:

- **`analysis/`**: Contém notebooks ou scripts utilizados para a análise exploratória dos dados (EDA) e visualizações iniciais.
- **`dataset/`**: Pasta destinada aos dados brutos do projeto (ex: `diabetes.csv`).
- **`models/`**: Contém a lógica de modelagem e processamento.
  - **`train_model.py`**: Script responsável por treinar os modelos de Machine Learning e salvar os melhores resultados.
  - **`pre_processor/`**: Módulo que contém a lógica de limpeza, imputação e normalização dos dados.
- **`main.py`**: O ponto de entrada da aplicação. Executa a interface de linha de comando para interação com o usuário.
- **`*.pkl`**: Arquivos binários que armazenam o modelo treinado (`model_diabetes.pkl`), o imputador (`imputer.pkl`) e o escalonador (`scaler.pkl`).
- **`Dockerfile`**: Arquivo de configuração para criação da imagem Docker do projeto.
- **`requirements.txt`**: Lista de dependências Python necessárias para rodar o projeto.

## 🐳 Como Rodar via Docker

Siga os passos abaixo para construir a imagem e executar o container:

1. **Construir a imagem:**
   No diretório raiz do projeto, execute:

   ```bash
   sudo docker build -t diabetes-app .
   ```

2. **Executar o container:**
   Para rodar a aplicação de forma interativa:
   ```bash
   sudo docker run -it diabetes-app
   ```

---

## 🚀 Como Rodar Localmente

Caso prefira rodar sem Docker, certifique-se de ter o Python instalado e siga estes passos:

1. **Instalar dependências:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Executar a aplicação:**
   ```bash
   python main.py
   ```
