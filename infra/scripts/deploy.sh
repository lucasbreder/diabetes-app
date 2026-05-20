#!/bin/bash
# ─────────────────────────────────────────────────────────────────
# deploy.sh - Script de deploy do Diabetes App na AWS
# Uso: ./infra/scripts/deploy.sh [environment]
# Exemplo: ./infra/scripts/deploy.sh production
# ─────────────────────────────────────────────────────────────────
set -e

ENVIRONMENT=${1:-production}
AWS_REGION="us-east-1"
APP_NAME="diabetes-app"

echo "=================================================="
echo " Deploy: $APP_NAME | Ambiente: $ENVIRONMENT"
echo "=================================================="

# 1. Verifica pré-requisitos
echo "[1/6] Verificando pré-requisitos..."
command -v docker  >/dev/null 2>&1 || { echo "Erro: Docker não instalado."; exit 1; }
command -v aws     >/dev/null 2>&1 || { echo "Erro: AWS CLI não instalado."; exit 1; }
command -v terraform >/dev/null 2>&1 || { echo "Erro: Terraform não instalado."; exit 1; }

# 2. Obtém URL do ECR
echo "[2/6] Obtendo URL do repositório ECR..."
ECR_URL=$(aws ecr describe-repositories \
  --repository-names $APP_NAME \
  --region $AWS_REGION \
  --query 'repositories[0].repositoryUri' \
  --output text 2>/dev/null || echo "")

if [ -z "$ECR_URL" ]; then
  echo "Repositório ECR não encontrado. Provisionando infra primeiro..."
  cd infra/terraform
  terraform init
  terraform apply -target=aws_ecr_repository.app -auto-approve
  ECR_URL=$(aws ecr describe-repositories \
    --repository-names $APP_NAME \
    --region $AWS_REGION \
    --query 'repositories[0].repositoryUri' \
    --output text)
  cd ../..
fi

echo "ECR: $ECR_URL"

# 3. Build da imagem Docker (treina o modelo durante o build)
echo "[3/6] Fazendo build da imagem Docker..."
docker build -t $APP_NAME:latest .

# 4. Push para o ECR
echo "[4/6] Enviando imagem para o ECR..."
aws ecr get-login-password --region $AWS_REGION | \
  docker login --username AWS --password-stdin $ECR_URL

docker tag $APP_NAME:latest $ECR_URL:latest
docker push $ECR_URL:latest
echo "Imagem enviada: $ECR_URL:latest"

# 5. Provisionar / atualizar infraestrutura com Terraform
echo "[5/6] Aplicando infraestrutura com Terraform..."
cd infra/terraform
terraform init -input=false
terraform plan -out=tfplan -var="environment=$ENVIRONMENT"
terraform apply tfplan
cd ../..

# 6. Forçar atualização das instâncias EC2 (rolling update)
echo "[6/6] Iniciando rolling update do Auto Scaling Group..."
ASG_NAME=$(terraform -chdir=infra/terraform output -raw autoscaling_group_name 2>/dev/null || echo "${APP_NAME}-asg")
aws autoscaling start-instance-refresh \
  --auto-scaling-group-name $ASG_NAME \
  --region $AWS_REGION \
  --preferences '{"MinHealthyPercentage": 50, "InstanceWarmup": 300}'

echo ""
echo "=================================================="
echo " Deploy concluído com sucesso!"
ALB_DNS=$(terraform -chdir=infra/terraform output -raw alb_dns_name 2>/dev/null || echo "(verifique no console AWS)")
echo " URL do app: https://$ALB_DNS"
echo "=================================================="
