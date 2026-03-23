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
- **`tests/`**: O projeto conta com testes automatizados utilizando o framework pytest.
  - **`test_dataset_pre_processor.py`**: Foco na validação do pipeline de pré-processamento de dados.

## 🚀 Como Iniciar (Menu Interativo)

Para facilitar a execução do projeto, foi criado um script gerenciador que permite escolher como quer rodar o projeto.

> [!IMPORTANT]
> **Pré-requisito (Ambiente Virtual)**
> Antes de executar o projeto, você **precisa** ativar o ambiente virtual (venv). Caso contrário, as dependências (como `pandas` e `scikit-learn`) não serão encontradas no seu sistema se você optar por rodar a demonstração localmente.

**No terminal, execute:**
```bash
# 1. Ative o ambiente virtual
source .venv/bin/activate

# 2. Rode o script do menu
python run.py
```

**Você verá um menu de opções para que você escolha algumas maneiras de executar o projeto:**

1. **Abrir interface web com Docker:** Inicia o Streamlit e o Ollama através do Docker Compose. O modelo Ollama llama3.2:1b será baixado automaticamente na primeira execução. 
2. **Uma vez iniciado o container, acesse em seu navegador o endereço para abrir o frontend do projeto:** `http://localhost:8501`.
3. **Rodar demonstração no terminal:** Mostra a simulação da IA diretamente no terminal.

## ▶️ Como executar os testes

  Certifique-se de que o ambiente virtual está ativado antes de rodar os testes.

  1. **Ativar ambiente virtual**
  source .venv/bin/activate

  2. **Executar todos os testes**
  pytest -v 
---
