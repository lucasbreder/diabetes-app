# ─────────────────────────────────────────────────────────────────
# LOAD BALANCER - Distribui tráfego entre instâncias do Diabetes App
# ─────────────────────────────────────────────────────────────────

resource "aws_lb" "app" {
  name               = "${var.app_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = data.aws_subnets.public.ids

  enable_deletion_protection = true

  access_logs {
    bucket  = aws_s3_bucket.logs.bucket
    prefix  = "alb-logs"
    enabled = true
  }

  tags = {
    Name        = "${var.app_name}-alb"
    Environment = var.environment
  }
}

# Listener HTTP → redireciona para HTTPS
resource "aws_lb_listener" "http_redirect" {
  load_balancer_arn = aws_lb.app.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type = "redirect"
    redirect {
      port        = "443"
      protocol    = "HTTPS"
      status_code = "HTTP_301"
    }
  }
}

# Listener HTTPS → encaminha para o Target Group (porta 8501)
resource "aws_lb_listener" "https" {
  load_balancer_arn = aws_lb.app.arn
  port              = "443"
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = aws_acm_certificate.app.arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

# Target Group: aponta para a porta 8501 (Streamlit)
resource "aws_lb_target_group" "app" {
  name     = "${var.app_name}-tg"
  port     = var.app_port  # 8501
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.default.id

  # Streamlit expõe /_stcore/health como endpoint de health check
  health_check {
    enabled             = true
    healthy_threshold   = 2
    unhealthy_threshold = 3
    timeout             = 10
    interval            = 30
    path                = "/_stcore/health"
    matcher             = "200"
  }

  tags = {
    Name        = "${var.app_name}-tg"
    Environment = var.environment
  }
}

# Certificado SSL via ACM
resource "aws_acm_certificate" "app" {
  domain_name       = "diabetes-app.com"
  validation_method = "DNS"

  subject_alternative_names = ["www.diabetes-app.com"]

  lifecycle {
    create_before_destroy = true
  }

  tags = {
    Name        = "${var.app_name}-cert"
    Environment = var.environment
  }
}

output "alb_dns_name" {
  description = "URL pública do Load Balancer"
  value       = aws_lb.app.dns_name
}
