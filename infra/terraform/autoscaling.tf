# ─────────────────────────────────────────────────────────────────
# AUTO SCALING - Diabetes App (Streamlit + Ollama)
# ─────────────────────────────────────────────────────────────────

# Launch Template: "molde" de cada instância EC2 criada
resource "aws_launch_template" "app" {
  name_prefix   = "${var.app_name}-lt-"
  image_id      = data.aws_ami.amazon_linux.id
  instance_type = var.instance_type

  iam_instance_profile {
    name = aws_iam_instance_profile.app.name
  }

  vpc_security_group_ids = [aws_security_group.app.id]

  monitoring {
    enabled = true # Habilita métricas detalhadas no CloudWatch (1 min)
  }

  # Script executado na inicialização de cada instância
  user_data = base64encode(<<-EOF
    #!/bin/bash
    set -e

    # Instala Docker
    yum update -y
    yum install -y docker
    systemctl enable docker
    systemctl start docker

    # Login no ECR e pull da imagem do app
    aws ecr get-login-password --region ${var.aws_region} | \
      docker login --username AWS --password-stdin ${aws_ecr_repository.app.repository_url}

    docker pull ${aws_ecr_repository.app.repository_url}:latest

    # Inicia Ollama (LLM local)
    docker run -d \
      --name ollama \
      --restart unless-stopped \
      -p ${var.ollama_port}:${var.ollama_port} \
      -v ollama_data:/root/.ollama \
      ollama/ollama:latest

    # Aguarda Ollama subir e baixa o modelo llama3.2:1b
    sleep 10
    docker exec ollama ollama pull llama3.2:1b

    # Inicia o app Streamlit (diabetes-app)
    docker run -d \
      --name diabetes-app \
      --restart unless-stopped \
      -p ${var.app_port}:${var.app_port} \
      -e OLLAMA_HOST=http://localhost:${var.ollama_port} \
      ${aws_ecr_repository.app.repository_url}:latest
  EOF
  )

  tag_specifications {
    resource_type = "instance"
    tags = {
      Name        = "${var.app_name}-instance"
      Environment = var.environment
    }
  }

  lifecycle {
    create_before_destroy = true
  }
}

# Auto Scaling Group: gerencia o conjunto de instâncias
resource "aws_autoscaling_group" "app" {
  name                = "${var.app_name}-asg"
  vpc_zone_identifier = data.aws_subnets.public.ids
  target_group_arns   = [aws_lb_target_group.app.arn]
  health_check_type   = "ELB"
  health_check_grace_period = 300 # 5 min para Ollama e modelo carregarem

  min_size         = var.min_capacity
  max_size         = var.max_capacity
  desired_capacity = var.desired_capacity

  launch_template {
    id      = aws_launch_template.app.id
    version = "$Latest"
  }

  tag {
    key                 = "Name"
    value               = "${var.app_name}-asg"
    propagate_at_launch = true
  }

  tag {
    key                 = "Environment"
    value               = var.environment
    propagate_at_launch = true
  }
}

# Política de Scale UP: CPU > 70% → adiciona 2 instâncias
resource "aws_autoscaling_policy" "scale_up" {
  name                   = "${var.app_name}-scale-up"
  autoscaling_group_name = aws_autoscaling_group.app.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = 2
  cooldown               = 300 # Escala no máximo a cada 5 minutos
}

# Política de Scale DOWN: CPU < 30% → remove 1 instância
resource "aws_autoscaling_policy" "scale_down" {
  name                   = "${var.app_name}-scale-down"
  autoscaling_group_name = aws_autoscaling_group.app.name
  adjustment_type        = "ChangeInCapacity"
  scaling_adjustment     = -1
  cooldown               = 300
}
