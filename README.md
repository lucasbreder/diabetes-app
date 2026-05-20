# Sistema de Diagnóstico de Diabetes

Este projeto utiliza Inteligência Artificial e técnicas avançadas de Machine Learning para auxiliar no diagnóstico de diabetes com base em dados clínicos, com foco em precisão, interpretabilidade e equidade.

## 📂 Organização dos Diretórios

- **`analysis/`**: Notebooks e scripts para Análise Exploratória de Dados (EDA).
- **`dataset/`**: Base de dados `diabetes.csv`.
- **`genetic_optimizer/`**: Implementação do motor de Algoritmo Genético para busca de hiperparâmetros.
- **`graphs/`**: Visualizações geradas (curvas de convergência, importância de variáveis, histogramas).
- **`models/`**:
  - **`train_model.py`**: Pipeline principal de treinamento e exportação do modelo.
- **`pre_processor/`**: Lógica de limpeza, imputação de valores ausentes e normalização.
- **`main.py`**: Interface de linha de comando para inferência.
- **`run.py`**: Menu interativo para execução simplificada.
- **`run_genetic_optimization.py`**: Script de experimentação e otimização avançada.
- **`*.pkl`**: Artefatos do modelo (modelo, imputer e scaler).

## 🚀 Como Iniciar

### Pré-requisitos
O projeto utiliza um ambiente virtual Python para gerenciar dependências.

```bash
# 1. Criar e ativar ambiente virtual
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate  # Windows

# 2. Instalar dependências
pip install -r requirements.txt
```

### Menu Interativo
Para facilitar o uso, utilize o script gerenciador:
```bash
python run.py
```
Este menu permite iniciar a interface web (Streamlit), rodar a demonstração no terminal ou verificar o status do ambiente.

---

## 🧠 Ciclo de Vida do Modelo

O projeto separa a fase de **Otimização Experimental** da fase de **Treinamento de Produção**.

### 1. Otimização via Algoritmo Genético (AG)
Antes de definir o modelo final, utilizamos Algoritmos Genéticos para encontrar os melhores hiperparâmetros, visando um equilíbrio entre performance e ética (equidade).

**O que é avaliado (Fitness Multi-objetivo):**
- **Recall (Sensibilidade):** Prioridade máxima para reduzir falsos negativos em saúde.
- **F1-Score:** Equilíbrio entre precisão e recall.
- **Especificidade:** Capacidade de identificar corretamente pacientes saudáveis.
- **Equidade (Equity):** Minimiza a disparidade de erro entre diferentes faixas etárias.

**Como rodar a otimização:**
```bash
python run_genetic_optimization.py
```
Este script executa 3 experimentos com diferentes populações e taxas de mutação, gerando relatórios comparativos e gráficos de convergência em `./graphs/`.

### 2. Treinamento e Exportação
Após identificar a melhor arquitetura, o script de treinamento consolida o modelo para uso na aplicação. O arquivo train_model já está configurado com os melhores hiperparâmetros do modelo vencedor, sendo assim bastar rodar o comando abaixo para salvar o modelo.

```bash
python -m models/train_model.py
```
**Ações realizadas:**
- Carrega e limpa os dados via `dataset_pre_processor`.
- Treina modelos baseline (KNN, Decision Tree, Logistic Regression) para comparação.
- Treina a **Random Forest** otimizada com tratamento de desbalanceamento de classes.
- Gera gráficos de **Feature Importance** em `./graphs/`.
- Salva os arquivos `model_diabetes.pkl`, `imputer.pkl` e `scaler.pkl`.

---

## 📊 Explicabilidade e Gráficos

A IA não deve ser uma "caixa preta". O projeto gera automaticamente:
- **Feature Importance:** Identifica quais fatores (Glicose, IMC, Idade) mais influenciam o diagnóstico.
- **Curvas de Convergência:** Mostram a evolução do Algoritmo Genético.
- **Gráficos de Comparação:** Confrontam o modelo baseline vs. o modelo otimizado pelo AG.

---

## 🐳 Docker e Interface Web

Para rodar a interface visual com suporte a LLM (Ollama) para explicações em linguagem natural:

```bash
docker-compose up --build
```
Acesse `http://localhost:8501` para interagir com o sistema.

## ▶️ Testes Automatizados

Garantimos a integridade do pipeline de dados com testes unitários:
```bash
pytest -v
```
Os testes validam o pré-processamento, garantindo que a normalização e a imputação de dados ocorram sem vazamento de informação (data leakage).