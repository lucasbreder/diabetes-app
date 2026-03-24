variable "aws_region" {
  description = "Região AWS onde os recursos serão criados"
  default     = "us-east-1"
}

variable "app_name" {
  description = "Nome do app (usado em tags e nomes de recursos)"
  default     = "diabetes-app"
}

variable "environment" {
  description = "Ambiente de execução"
  default     = "production"
}

# ─── Auto Scaling ───────────────────────────────────────────────
variable "min_capacity" {
  description = "Número mínimo de instâncias EC2"
  default     = 2
}

variable "max_capacity" {
  description = "Número máximo de instâncias EC2"
  default     = 10
}

variable "desired_capacity" {
  description = "Número inicial de instâncias EC2"
  default     = 2
}

variable "instance_type" {
  description = "Tipo de instância EC2 (t3.medium tem 2 vCPU e 4GB RAM - adequado para Streamlit + Ollama)"
  default     = "t3.large"
}

# ─── Portas da Aplicação ─────────────────────────────────────────
variable "app_port" {
  description = "Porta em que o Streamlit roda"
  default     = 8501
}

variable "ollama_port" {
  description = "Porta em que o Ollama (LLM) roda"
  default     = 11434
}

# ─── Alertas ────────────────────────────────────────────────────
variable "alert_email" {
  description = "Email para receber alertas do CloudWatch"
  default     = "devops@diabetes-app.com"
}

variable "cpu_scale_up_threshold" {
  description = "% de CPU para disparar scale up"
  default     = 70
}

variable "cpu_scale_down_threshold" {
  description = "% de CPU para disparar scale down"
  default     = 30
}
