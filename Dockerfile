# 1. Imagem Base
FROM python:3.11-slim

# 2. Configurações de Ambiente
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# 3. Define a pasta de trabalho
WORKDIR /app

# 4. Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copia o projeto e Roda o Treinamento do Modelo
COPY . .
RUN python -m models.train_model

EXPOSE 8501

# 6. Comando padrão
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]