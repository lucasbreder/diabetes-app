"""
Assistente Médica Especializada em Saúde da Mulher
Pipeline LangChain com integração a bases de dados clínicas especializadas.
"""

from .pipeline import AssistenteMedico, EventoChat, criar_pipeline_assistente
from .database import init_db
from .safety import REGRAS_SEGURANCA_PROMPT, aplicar_guardrails_resposta
from .validation_pipeline import processar_resposta_final
from .audit import relatorio_resumo_auditoria
from .security_protocols import verificar_identidade_profissional

__all__ = [
    "AssistenteMedico",
    "EventoChat",
    "criar_pipeline_assistente",
    "init_db",
    "REGRAS_SEGURANCA_PROMPT",
    "aplicar_guardrails_resposta",
    "processar_resposta_final",
    "relatorio_resumo_auditoria",
    "verificar_identidade_profissional",
]
