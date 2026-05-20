# ─────────────────────────────────────────────────────────────────
# MONITORAMENTO - CloudWatch para o Diabetes App
# ─────────────────────────────────────────────────────────────────

# Dashboard com métricas do app
resource "aws_cloudwatch_dashboard" "app" {
  dashboard_name = "${var.app_name}-dashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          title  = "CPU das Instâncias EC2"
          metrics = [["AWS/EC2", "CPUUtilization", "AutoScalingGroupName", aws_autoscaling_group.app.name]]
          period = 60
          stat   = "Average"
          view   = "timeSeries"
          yAxis  = { left = { min = 0, max = 100 } }
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          title  = "Requisições no Load Balancer"
          metrics = [["AWS/ApplicationELB", "RequestCount", "LoadBalancer", aws_lb.app.arn_suffix]]
          period = 60
          stat   = "Sum"
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          title  = "Tempo de Resposta do Streamlit (ms)"
          metrics = [["AWS/ApplicationELB", "TargetResponseTime", "LoadBalancer", aws_lb.app.arn_suffix]]
          period = 60
          stat   = "Average"
          view   = "timeSeries"
        }
      },
      {
        type   = "metric"
        width  = 12
        height = 6
        properties = {
          title  = "Instâncias em Execução"
          metrics = [["AWS/AutoScaling", "GroupInServiceInstances", "AutoScalingGroupName", aws_autoscaling_group.app.name]]
          period = 60
          stat   = "Average"
          view   = "timeSeries"
        }
      }
    ]
  })
}

# ─── Alarmes de CPU ────────────────────────────────────────────────

resource "aws_cloudwatch_metric_alarm" "cpu_high" {
  alarm_name          = "${var.app_name}-cpu-high"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2           # 2 períodos consecutivos
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 120         # janela de 2 minutos
  statistic           = "Average"
  threshold           = var.cpu_scale_up_threshold  # 70%
  alarm_description   = "CPU acima de ${var.cpu_scale_up_threshold}%: disparando scale up"

  alarm_actions = [
    aws_autoscaling_policy.scale_up.arn,
    aws_sns_topic.alerts.arn
  ]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.app.name
  }
}

resource "aws_cloudwatch_metric_alarm" "cpu_low" {
  alarm_name          = "${var.app_name}-cpu-low"
  comparison_operator = "LessThanThreshold"
  evaluation_periods  = 2
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = 120
  statistic           = "Average"
  threshold           = var.cpu_scale_down_threshold  # 30%
  alarm_description   = "CPU abaixo de ${var.cpu_scale_down_threshold}%: disparando scale down"

  alarm_actions = [aws_autoscaling_policy.scale_down.arn]

  dimensions = {
    AutoScalingGroupName = aws_autoscaling_group.app.name
  }
}

# Alarme: instâncias com falha no health check
resource "aws_cloudwatch_metric_alarm" "unhealthy_hosts" {
  alarm_name          = "${var.app_name}-unhealthy-hosts"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "UnHealthyHostCount"
  namespace           = "AWS/ApplicationELB"
  period              = 60
  statistic           = "Average"
  threshold           = 0
  alarm_description   = "Existe pelo menos uma instância sem responder ao health check"

  alarm_actions = [aws_sns_topic.alerts.arn]

  dimensions = {
    TargetGroup  = aws_lb_target_group.app.arn_suffix
    LoadBalancer = aws_lb.app.arn_suffix
  }
}

# ─── SNS: Notificações por Email ──────────────────────────────────

resource "aws_sns_topic" "alerts" {
  name = "${var.app_name}-alerts"
}

resource "aws_sns_topic_subscription" "email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

# ─── Logs Centralizados ───────────────────────────────────────────

# Log Group do app Streamlit
resource "aws_cloudwatch_log_group" "app" {
  name              = "/diabetes-app/streamlit"
  retention_in_days = 30

  tags = {
    Environment = var.environment
    Application = var.app_name
  }
}

# Log Group do Ollama (LLM)
resource "aws_cloudwatch_log_group" "ollama" {
  name              = "/diabetes-app/ollama"
  retention_in_days = 14

  tags = {
    Environment = var.environment
    Application = "${var.app_name}-ollama"
  }
}

# Log Group das interações LLM (espelha o logs/llm_interactions.jsonl local)
resource "aws_cloudwatch_log_group" "llm_interactions" {
  name              = "/diabetes-app/llm-interactions"
  retention_in_days = 90

  tags = {
    Environment = var.environment
    Application = "${var.app_name}-llm"
  }
}
