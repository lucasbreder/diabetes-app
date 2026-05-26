# Sistema de Diagnóstico de Diabetes + Assistente Médico de Saúde da Mulher

Este projeto utiliza Inteligência Artificial e técnicas avançadas de Machine Learning para auxiliar no diagnóstico de diabetes com base em dados clínicos. Além disso, conta com um **Assistente Médico especializado em Saúde da Mulher**, integrando LLM local (Ollama) com protocolos clínicos de ginecologia, obstetrícia e saúde reprodutiva.

## 📂 Organização dos Diretórios

- **`analysis/`**: Notebooks e scripts para Análise Exploratória de Dados (EDA).
- **`dataset/`**: Base de dados `diabetes.csv` e scripts de geração/junção de datasets fictícios.
- **`genetic_optimizer/`**: Implementação do motor de Algoritmo Genético para busca de hiperparâmetros.
- **`graphs/`**: Visualizações geradas (curvas de convergência, importância de variáveis, histogramas).
- **`medical_assistant/`**: Módulo do Assistente Médico de Saúde da Mulher (ver seção abaixo).
- **`models/`**:
  - **`train_model.py`**: Pipeline principal de treinamento e exportação do modelo.
- **`pages/`**:
  - **`assistente_medico.py`**: Interface Streamlit multitarefa do assistente médico.
- **`pre_processor/`**: Lógica de limpeza, imputação de valores ausentes e normalização.
- **`main.py`**: Interface de linha de comando para inferência.
- **`run.py`**: Menu interativo para execução simplificada.
- **`run_genetic_optimization.py`**: Script de experimentação e otimização avançada.
- **`*.pkl`**: Artefatos do modelo (modelo, imputer e scaler).

---

## 🏥 Assistente Médico — Saúde da Mulher

Módulo completo de suporte clínico integrando LLM local (Ollama) com banco de dados de protocolos médicos, prontuários eletrônicos e rastreamento preventivo.

### Funcionalidades

| Aba | Descrição |
|-----|-----------|
| **Chat Clínico** | Chat contextualizado com dados da paciente, histórico obstétrico e protocolos FEBRASGO/MS |
| **Triagem de Sintomas** | Classificação de urgência (verde/amarelo/laranja/vermelho) com recomendações |
| **Alertas Preventivos** | Exames em atraso e resultados alterados com plano de ação personalizado |
| **Triagem de Violência Doméstica** | Instrumento WAST (8 itens) com cálculo de risco e relatório confidencial |
| **Encaminhamentos** | Sugestões multidisciplinares com prioridade (imediato/urgente/eletivo) e orientações pós-consulta |

### Estrutura do módulo `medical_assistant/`

```
medical_assistant/
├── __init__.py
├── database.py          # CRUD SQLAlchemy com SQLite
├── models.py            # ORM (Paciente, Prontuário, Exames, Ciclos, Medicamentos, Protocolos)
├── pipeline.py          # Pipeline LCEL principal — AssistenteMedico com histórico de chat
├── seed_data.py         # 10 protocolos FEBRASGO/MS, 8 medicamentos, 3 pacientes demo
├── tools.py             # LangChain Tools (para futura extensão ReAct)
└── chains/
    ├── __init__.py
    ├── alerts.py        # Chain de alertas de exames preventivos
    ├── dv_screening.py  # Chain de triagem de violência doméstica (WAST)
    ├── referrals.py     # Chain de encaminhamentos e orientações pós-consulta
    └── triage.py        # Chain de triagem de sintomas
```

### Pacientes demo

| Paciente | Perfil clínico |
|----------|---------------|
| Ana Clara Ferreira | Suspeita de SOP, 28 anos |
| Beatriz Santos Lima | Pós-menopausa, 55 anos |
| Carla Oliveira Nascimento | Suspeita de endometriose, 34 anos |

### Stack técnica do assistente

- **LangChain ≥ 0.3 / LCEL** — cadeias compostas com `PromptTemplate | OllamaLLM | StrOutputParser()`
- **langchain-ollama** — `OllamaLLM` (usa `/api/generate`, compatível com instalações sem `/api/chat`)
- **SQLAlchemy 2.0 + SQLite** — `expire_on_commit=False` para evitar `DetachedInstanceError`
- **Streamlit** — interface multipage com streaming de respostas

### Pré-requisitos adicionais

```bash
# Ollama instalado e rodando com o modelo llama3:latest
ollama pull llama3

# Dependências Python
pip install langchain langchain-community langchain-ollama langchain-core sqlalchemy
```

### Iniciar o assistente médico

```bash
# Terminal 1 — Ollama
ollama serve

# Terminal 2 — Streamlit
streamlit run app.py
```

# PIN para acesso às áreas restritas
saudemulher2026

Acesse `http://localhost:8501` → aba **Assistente Médico**.

---

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



** Lourenço 19/05/2026 **
Foi criado um script para gerar um dataset ficticio com o objetivo de atender o solicitado no item abaixo do techchalenge.
Foi criado outro script para unir os datos ficticios e o dataset que utilizamos como base inicial do projeto. Tudo está em /dataset

  Requisitos obrigatórios >> Entregas técnicas >> Protocolos médicos especializados:
    • Protocolos de atendimento ginecológico e obstétrico do hospital;
    • Diretrizes para identificação e manejo de violência doméstica;
    • Protocolos de triagem para câncer de mama e colo do útero;
    • Procedimentos para emergências obstétricas;
    • Diretrizes para saúde mental da mulher (depressão pós-parto,
    ansiedade etc.);

Com esses scripts finalizamos a entrega o Item 1.Fine-tuning de LLM com dados médicos especializados em saúde da mulher
