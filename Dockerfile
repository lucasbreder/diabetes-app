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

# 5. Copia o projeto
COPY . .
RUN python -m models.train_model

# 6. Comando padrão
CMD ["python", "main.py"]